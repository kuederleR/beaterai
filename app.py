import sys
import os
import time
import traceback
import threading
import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# Flush all prints immediately so docker logs shows them in real time
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
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', '0')
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass

DEVICE_STR = os.environ.get('DEVICE', '0')
IMG_SIZE = 640
CONF_THRES = 0.3
IOU_THRES = 0.45

# --- Shared state for background inference thread ---
latest_frame = None
frame_lock = threading.Lock()
status_info = {"state": "initializing", "error": None, "frames_processed": 0, "fps": 0.0}


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
    """Create a JPEG-encoded error frame with the given message."""
    error_img = np.zeros((480, 800, 3), dtype=np.uint8)
    # Word-wrap long messages
    words = message.split(' ')
    lines, line = [], ''
    for w in words:
        if len(line + ' ' + w) > 60:
            lines.append(line)
            line = w
        else:
            line = (line + ' ' + w).strip()
    lines.append(line)
    for i, l in enumerate(lines):
        cv2.putText(error_img, l, (20, 200 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    _, buf = cv2.imencode('.jpg', error_img)
    return buf.tobytes()


def inference_loop(model, device, half):
    """Background thread: reads video, runs inference, stores latest JPEG frame."""
    global latest_frame, status_info
    import av

    # Diagnostics: check file exists
    if isinstance(VIDEO_SOURCE, str):
        if os.path.exists(VIDEO_SOURCE):
            size = os.path.getsize(VIDEO_SOURCE)
            print(f"[inference] Video file exists: {VIDEO_SOURCE} ({size} bytes)", flush=True)
            if size < 1000:
                print(f"[inference] WARNING: File is suspiciously small!", flush=True)
        else:
            msg = f"Video file does not exist: {VIDEO_SOURCE}"
            print(f"[inference] ERROR: {msg}", flush=True)
            status_info["state"] = "error"
            status_info["error"] = msg
            with frame_lock:
                latest_frame = make_error_frame(msg)
            return

    print(f"[inference] Opening video source with PyAV: {VIDEO_SOURCE}", flush=True)

    try:
        container = av.open(str(VIDEO_SOURCE))
    except Exception as e:
        msg = f"PyAV could not open {VIDEO_SOURCE}: {e}"
        print(f"[inference] ERROR: {msg}", flush=True)
        status_info["state"] = "error"
        status_info["error"] = msg
        with frame_lock:
            latest_frame = make_error_frame(msg)
        return

    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"
    print(f"[inference] Video opened: {video_stream.codec_context.width}x"
          f"{video_stream.codec_context.height}, codec={video_stream.codec_context.name}",
          flush=True)

    fps_counter = 0
    fps_start = time.time()
    status_info["state"] = "running"
    print("[inference] Starting inference loop...", flush=True)

    while True:
        try:
            for frame_av in container.decode(video=0):
                # Convert PyAV frame to numpy BGR array (OpenCV format)
                frame = frame_av.to_ndarray(format='bgr24')

                im0 = frame.copy()
                h, w = im0.shape[:2]

                # Preprocess
                img, ratio, pad = letterbox(im0, IMG_SIZE, stride=32, auto=True)
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                img = np.ascontiguousarray(img)

                img_tensor = torch.from_numpy(img).to(device)
                img_tensor = img_tensor.half() if half else img_tensor.float()
                img_tensor /= 255.0
                if img_tensor.ndimension() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                # Inference
                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = model(img_tensor)
                    pred = split_for_trace_model(pred, anchor_grid)
                    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)
                    # YOLOPv2's mask functions hardcode extraction from a 384x640 padded tensor
                    da_seg_mask = driving_area_mask(seg)
                    ll_seg_mask = lane_line_mask(ll)

                # --- Draw segmentation overlays ---
                da_mask = da_seg_mask
                ll_mask = ll_seg_mask

                if isinstance(da_mask, torch.Tensor):
                    da_mask = da_mask.cpu().numpy()
                if isinstance(ll_mask, torch.Tensor):
                    ll_mask = ll_mask.cpu().numpy()

                # YOLOPv2 hardcodes the mask output to 720x1280, so we create a canvas of that size
                color_area = np.zeros((da_mask.shape[0], da_mask.shape[1], 3), dtype=np.uint8)
                color_area[da_mask == 1] = [0, 255, 0]   # Drivable area: Green
                color_area[ll_mask == 1] = [0, 0, 255]   # Lane lines: Red (BGR)

                # Resize the 720x1280 mask to the actual video frame dimensions
                color_area = cv2.resize(color_area, (w, h), interpolation=cv2.INTER_NEAREST)

                # Alpha-blend the segmentation mask onto the original frame
                mask = np.any(color_area != 0, axis=-1)
                im0[mask] = cv2.addWeighted(im0, 0.5, color_area, 0.5, 0)[mask]

                # --- Draw detection boxes ---
                for det in pred:
                    if len(det):
                        det[:, :4] = scale_coords(
                            img_tensor.shape[2:], det[:, :4], im0.shape
                        ).round()
                        for *xyxy, conf, cls in reversed(det):
                            # Yellow bounding boxes (BGR format)
                            plot_one_box(xyxy, im0, color=(0, 255, 255),
                                         line_thickness=3)

                # Encode to JPEG and store
                _, buf = cv2.imencode('.jpg', im0,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                with frame_lock:
                    latest_frame = buf.tobytes()

                fps_counter += 1
                elapsed = time.time() - fps_start
                if elapsed >= 2.0:
                    status_info["fps"] = round(fps_counter / elapsed, 1)
                    fps_counter = 0
                    fps_start = time.time()
                status_info["frames_processed"] += 1

            # Video ended — loop by re-opening
            print("[inference] Video ended, looping...", flush=True)
            container.close()
            container = av.open(str(VIDEO_SOURCE))
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"

        except Exception:
            traceback.print_exc()
            status_info["state"] = "error"
            status_info["error"] = traceback.format_exc()
            with frame_lock:
                latest_frame = make_error_frame(
                    "Inference crashed - see docker logs"
                )
            time.sleep(5)
            continue


def generate_mjpeg():
    """Yield MJPEG frames from the shared latest_frame buffer."""
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            # Model still loading / no frame yet – send a "please wait" image
            wait_img = np.zeros((480, 800, 3), dtype=np.uint8)
            cv2.putText(wait_img, "Initializing model, please wait...",
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            _, buf = cv2.imencode('.jpg', wait_img)
            frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 fps cap to avoid overwhelming the network


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string('''
    <html>
        <head>
            <title>YOLOPv2 Stream - Jetson Orin Nano</title>
            <style>
                body { background-color: #1a1a1a; color: white; display: flex;
                       flex-direction: column; align-items: center;
                       font-family: 'Inter', sans-serif; margin: 0; }
                h1 { margin-top: 20px; font-weight: 300; }
                .video-container {
                    margin-top: 20px; padding: 10px;
                    background-color: #2a2a2a; border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
                }
                img { max-width: 100%; border-radius: 8px; }
                .status { margin-top: 12px; font-size: 14px; color: #aaa; }
            </style>
        </head>
        <body>
            <h1>YOLOPv2 Live Inference</h1>
            <div class="video-container">
                <img src="/video_feed">
            </div>
            <div class="status" id="status">Loading...</div>
            <script>
                setInterval(async () => {
                    try {
                        const r = await fetch('/status');
                        const d = await r.json();
                        document.getElementById('status').innerText =
                            `State: ${d.state} | FPS: ${d.fps} | Frames: ${d.frames_processed}`;
                    } catch(e) {}
                }, 2000);
            </script>
        </body>
    </html>
    ''')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(status_info)


# --- Startup ---
if __name__ == '__main__':
    print("=" * 50, flush=True)
    print("Loading YOLOPv2 model...", flush=True)

    if not os.path.exists(WEIGHTS_PATH):
        print("Weights not found, downloading...", flush=True)
        os.makedirs('data/weights', exist_ok=True)
        os.system(
            f"wget -q -O {WEIGHTS_PATH} "
            "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
        )

    device = select_device(DEVICE_STR)
    half = device.type != 'cpu'

    model = torch.jit.load(WEIGHTS_PATH)
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # Warmup
    print("Warming up model...", flush=True)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        if half:
            dummy = dummy.half()
        model(dummy)

    print("Model ready. Starting inference thread...", flush=True)

    # Start background inference thread
    t = threading.Thread(target=inference_loop, args=(model, device, half),
                         daemon=True)
    t.start()

    print("Starting Flask server on port 5000...", flush=True)
    print("=" * 50, flush=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
