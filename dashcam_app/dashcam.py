import sys
import os
import time
import threading
import datetime
import cv2
import torch
import numpy as np
from flask import Flask, Response, jsonify, request, render_template

os.environ['PYTHONUNBUFFERED'] = '1'

# Append current directory to path so we can import YOLOPv2 utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import (
    select_device, scale_coords, non_max_suppression,
    split_for_trace_model, driving_area_mask, lane_line_mask,
    plot_one_box
)

app = Flask(__name__)

# --- Configuration ---
WEIGHTS_PATH = 'data/weights/yolopv2.pt'
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', '/dev/video0')
DEVICE_STR = os.environ.get('DEVICE', '0')
IMG_SIZE = 640
CONF_THRES = 0.3
IOU_THRES = 0.45
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 800
TARGET_FPS = 30

# --- State ---
latest_web_frame = None
raw_frame_buffer = None
frame_lock = threading.Lock()
state = {
    "recording": False,
    "overlay_enabled": False,
    "capture_fps": 0.0,
    "web_fps": 0.0,
    "recording_since": None,
    "error": None
}
video_writer = None

# --- Utilities ---
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def make_error_frame(message):
    error_img = np.zeros((480, 800, 3), dtype=np.uint8)
    cv2.putText(error_img, message, (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    _, buf = cv2.imencode('.jpg', error_img)
    return buf.tobytes()

# --- Thread 1: Camera Capture and Direct Recording ---
def capture_loop():
    global raw_frame_buffer, state, video_writer, latest_web_frame
    
    # Use V4L2 backend explicitly to request MJPEG from the USB camera
    print(f"[INFO] Opening camera {VIDEO_SOURCE} via V4L2 with MJPEG format...", flush=True)
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    if not cap.isOpened():
        state["error"] = f"Could not open camera {VIDEO_SOURCE}"
        print(f"[ERROR] {state['error']}", flush=True)
        with frame_lock:
            latest_web_frame = make_error_frame(state["error"])
        return

    fps_counter = 0
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Could not read frame from camera. Retrying...", flush=True)
            time.sleep(0.01)
            continue
            
        with frame_lock:
            # Store copy in shared buffer for inference thread
            raw_frame_buffer = frame.copy()
        
        # Recording logic (raw frame, completely decoupled from inference)
        if state["recording"]:
            if video_writer is None:
                os.makedirs("recordings", exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/dashcam_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(filename, fourcc, TARGET_FPS, (frame.shape[1], frame.shape[0]))
                state["recording_since"] = time.time()
                print(f"[INFO] Started recording: {filename}", flush=True)
            video_writer.write(frame)
        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                state["recording_since"] = None
                print(f"[INFO] Stopped recording", flush=True)

        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 2.0:
            state["capture_fps"] = round(fps_counter / elapsed, 1)
            fps_counter = 0
            fps_start = time.time()


# --- Thread 2: AI Inference and Web Encoding ---
def inference_loop(model, device, half):
    global latest_web_frame, raw_frame_buffer, state
    
    fps_counter = 0
    fps_start = time.time()

    while True:
        with frame_lock:
            if raw_frame_buffer is None:
                time.sleep(0.01)
                continue
            im0 = raw_frame_buffer.copy()
            # Clear buffer to ensure we don't process the exact same frame twice if inference is very fast
            raw_frame_buffer = None

        # Overlay logic
        if state["overlay_enabled"] and model is not None:
            h, w = im0.shape[:2]
            img, ratio, pad = letterbox(im0, IMG_SIZE, stride=32, auto=True)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)

            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.half() if half else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                [pred, anchor_grid], seg, ll = model(img_tensor)
                pred = split_for_trace_model(pred, anchor_grid)
                pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)
                da_seg_mask = driving_area_mask(seg)
                ll_seg_mask = lane_line_mask(ll)

            # --- Draw segmentation overlays ---
            da_mask = da_seg_mask.cpu().numpy() if isinstance(da_seg_mask, torch.Tensor) else da_seg_mask
            ll_mask = ll_seg_mask.cpu().numpy() if isinstance(ll_seg_mask, torch.Tensor) else ll_seg_mask

            color_area = np.zeros((da_mask.shape[0], da_mask.shape[1], 3), dtype=np.uint8)
            color_area[da_mask == 1] = [0, 255, 0]
            color_area[ll_mask == 1] = [0, 0, 255]

            color_area = cv2.resize(color_area, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.any(color_area != 0, axis=-1)
            im0[mask] = cv2.addWeighted(im0, 0.5, color_area, 0.5, 0)[mask]

            # --- Draw detection boxes ---
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(
                        img_tensor.shape[2:], det[:, :4], im0.shape
                    ).round()
                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, im0, color=(0, 255, 255), line_thickness=3)
            
            # Help garbage collection since GPU memory is precious
            del img_tensor, pred, anchor_grid, seg, ll

        # Encode to JPEG for the web interface
        _, buf = cv2.imencode('.jpg', im0, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_web_frame = buf.tobytes()

        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 2.0:
            state["web_fps"] = round(fps_counter / elapsed, 1)
            fps_counter = 0
            fps_start = time.time()


def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_web_frame
        if frame is None:
            wait_img = np.zeros((480, 800, 3), dtype=np.uint8)
            cv2.putText(wait_img, "Initializing dashcam...",
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            _, buf = cv2.imencode('.jpg', wait_img)
            frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    duration = 0
    if state["recording"] and state["recording_since"]:
        duration = int(time.time() - state["recording_since"])
    
    return jsonify({
        "capture_fps": state["capture_fps"],
        "web_fps": state["web_fps"],
        "recording": state["recording"],
        "overlay_enabled": state["overlay_enabled"],
        "recording_duration": duration,
        "error": state["error"]
    })

@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    data = request.json
    if "enabled" in data:
        state["recording"] = data["enabled"]
    else:
        state["recording"] = not state["recording"]
    return jsonify({"success": True, "recording": state["recording"]})

@app.route('/api/toggle_overlay', methods=['POST'])
def toggle_overlay():
    data = request.json
    if "enabled" in data:
        state["overlay_enabled"] = data["enabled"]
    else:
        state["overlay_enabled"] = not state["overlay_enabled"]
    return jsonify({"success": True, "overlay_enabled": state["overlay_enabled"]})

if __name__ == '__main__':
    print("=" * 50, flush=True)
    
    if not os.path.exists(WEIGHTS_PATH):
        print("Weights not found, downloading...", flush=True)
        os.makedirs('data/weights', exist_ok=True)
        os.system(
            f"wget -q -O {WEIGHTS_PATH} "
            "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
        )

    device = select_device(DEVICE_STR)
    half = device.type != 'cpu'

    model = None
    if os.path.exists(WEIGHTS_PATH):
        print("Loading YOLOpV2 model...", flush=True)
        try:
            model = torch.jit.load(WEIGHTS_PATH)
            model = model.to(device)
            if half:
                model.half()
            model.eval()
            print("Warming up model...", flush=True)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
                if half:
                    dummy = dummy.half()
                model(dummy)
            print("Model ready.", flush=True)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)

    # Start multi-threaded architecture
    print("Starting Capture thread...", flush=True)
    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_capture.start()

    print("Starting Inference thread...", flush=True)
    t_inference = threading.Thread(target=inference_loop, args=(model, device, half), daemon=True)
    t_inference.start()

    print("Starting Flask server on port 5001...", flush=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
