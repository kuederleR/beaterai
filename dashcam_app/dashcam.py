import sys
import os
import logging
import time
import threading
import datetime
import cv2
import numpy as np
import torch
import json
from scipy.ndimage import gaussian_filter1d

# --- Fix for NVIDIA container headless OpenCV ---
if not hasattr(cv2, 'imshow'):
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: None
    cv2.destroyAllWindows = lambda *args, **kwargs: None
if not hasattr(cv2, 'IMREAD_COLOR'):
    cv2.IMREAD_COLOR = 1
if not hasattr(cv2, 'IMREAD_GRAYSCALE'):
    cv2.IMREAD_GRAYSCALE = 0
if not hasattr(cv2, 'IMREAD_UNCHANGED'):
    cv2.IMREAD_UNCHANGED = -1

from flask import Flask, Response, jsonify, request, render_template
from ultralytics import YOLO

# Import TwinLiteNet for fast drivable area segmentation
from twinlite_detector import TwinLiteDetector

os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)

# --- Configuration ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', '/dev/video0')
DEV_VIDEO_PATH = os.environ.get('DEV_VIDEO_PATH', None)
DEVICE_STR = os.environ.get('DEVICE', '0')
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 800
TARGET_FPS = 30

INFER_WIDTH = 640
INFER_HEIGHT = 400

# FCW Settings
FCW_WARNING_WIDTH = 200  # If a car's bounding box is wider than this in pixels, it's very close
CENTER_LANE_X_MIN = INFER_WIDTH // 3
CENTER_LANE_X_MAX = (INFER_WIDTH // 3) * 2

# LDW Settings
LDW_MAX_DRIFT = 40 # Max pixel drift from center before warning

# --- State ---
latest_web_frame = None
raw_frame_buffer = None
frame_lock = threading.Lock()
state = {
    "recording": False,
    "adas_enabled": False,
    "fcw_warning": False,
    "ldw_warning": False,
    "capture_fps": 0.0,
    "web_fps": 0.0,
    "recording_since": None,
    "error": None,
    "cuda_available": False,
    "gpu_device_name": "None",
    "yolo_device": "Unknown",
    "twinlite_device": "Unknown",
    "left_poly_history": [],
    "right_poly_history": []
}
video_writer = None

def make_error_frame(message):
    error_img = np.zeros((480, 800, 3), dtype=np.uint8)
    cv2.putText(error_img, message, (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    _, buf = cv2.imencode('.jpg', error_img)
    return buf.tobytes()

# --- Thread 1: Camera Capture and Direct Recording ---
def capture_loop():
    global raw_frame_buffer, state, video_writer, latest_web_frame
    
    if DEV_VIDEO_PATH and os.path.exists(DEV_VIDEO_PATH):
        print(f"[INFO] DEV MODE: Looping video from {DEV_VIDEO_PATH}", flush=True)
        cap = cv2.VideoCapture(DEV_VIDEO_PATH)
    else:
        # Use V4L2 backend explicitly to request MJPEG from the USB camera
        print(f"[INFO] Opening camera {VIDEO_SOURCE} via V4L2 with MJPEG format...", flush=True)
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    if not cap.isOpened():
        state["error"] = f"Could not open camera {VIDEO_SOURCE if not DEV_VIDEO_PATH else DEV_VIDEO_PATH}"
        print(f"[ERROR] {state['error']}", flush=True)
        with frame_lock:
            latest_web_frame = make_error_frame(state["error"])
        return

    fps_counter = 0
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if DEV_VIDEO_PATH:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            print("[WARNING] Could not read frame from camera. Retrying...", flush=True)
            time.sleep(0.01)
            continue
            
        if DEV_VIDEO_PATH:
            # Simulate real-time framerate for video files so it doesn't run at 1000fps
            time.sleep(1.0 / TARGET_FPS)
            
        with frame_lock:
            raw_frame_buffer = frame.copy()
        
        # Recording logic
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
def smooth_spline(fitx, history, max_history=5):
    """Applies a moving average to smooth out lane splines across frames."""
    if fitx is None:
        if len(history) > 0:
            return np.mean(history, axis=0)
        return None
        
    history.append(fitx)
    if len(history) > max_history:
        history.pop(0)
        
    return np.mean(history, axis=0)

def extract_lane_direct(ll_mask, car_center_x, calibrated_width=None):
    h, w = ll_mask.shape
    
    # We trace from the hood up to the horizon
    start_y = h - 20
    end_y = 150
    
    y_range = np.arange(start_y, end_y - 1, -1)
    left_x = np.full_like(y_range, np.nan, dtype=np.float32)
    right_x = np.full_like(y_range, np.nan, dtype=np.float32)
    
    center = car_center_x
    
    for i, y in enumerate(y_range):
        row = ll_mask[y, :]
        
        # Search outwards from the current center
        left_half = row[:int(center)]
        right_half = row[int(center):]
        
        l_idx = np.where(left_half > 0)[0]
        r_idx = np.where(right_half > 0)[0]
        
        found_l = False
        found_r = False
        
        if len(l_idx) > 0:
            left_x[i] = l_idx[-1]
            found_l = True
            
        if len(r_idx) > 0:
            right_x[i] = r_idx[0] + int(center)
            found_r = True
            
        if found_l and found_r:
            center = (left_x[i] + right_x[i]) / 2.0
        elif found_l and calibrated_width is not None and y < len(calibrated_width):
            # Ghost right
            right_x[i] = left_x[i] + calibrated_width[y]
            center = left_x[i] + calibrated_width[y] / 2.0
        elif found_r and calibrated_width is not None and y < len(calibrated_width):
            # Ghost left
            left_x[i] = right_x[i] - calibrated_width[y]
            center = right_x[i] - calibrated_width[y] / 2.0
            
    # Remove NaN values (rows where both lines were missing)
    valid_idx = ~np.isnan(left_x) & ~np.isnan(right_x)
    if not np.any(valid_idx):
        return None, None, None, start_y
        
    valid_y = y_range[valid_idx]
    valid_left = left_x[valid_idx]
    valid_right = right_x[valid_idx]
    
    # Smooth the extracted path to remove pixel jitters
    if len(valid_y) > 5:
        from scipy.ndimage import gaussian_filter1d
        valid_left = gaussian_filter1d(valid_left, sigma=3.0)
        valid_right = gaussian_filter1d(valid_right, sigma=3.0)
        
    return valid_left, valid_right, valid_y, start_y

def inference_loop():
    global latest_web_frame, raw_frame_buffer, state
    
    fps_counter = 0
    fps_start = time.time()

    print("[INFO] Loading YOLOv8n object detector...", flush=True)
    os.makedirs('models', exist_ok=True)
    yolo_model = YOLO('yolov8n.pt') # Will auto-download if missing
    
    state["cuda_available"] = torch.cuda.is_available()
    state["gpu_device_name"] = torch.cuda.get_device_name(0) if state["cuda_available"] else "None"
    
    if state["cuda_available"]:
        engine_path = 'models/yolov8n.engine'
        if not os.path.exists(engine_path):
            print("[INFO] Exporting YOLOv8n to TensorRT engine (this will take a few minutes)...", flush=True)
            try:
                yolo_model.export(format='engine', device='0', half=True)
                if os.path.exists('yolov8n.engine'):
                    os.rename('yolov8n.engine', engine_path)
            except Exception as e:
                print(f"[ERROR] Failed to export TensorRT: {e}", flush=True)
                
        if os.path.exists(engine_path):
            print("[INFO] Loading TensorRT engine from cache...", flush=True)
            yolo_model = YOLO(engine_path, task='detect')
    
    yolo_dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    state["yolo_device"] = yolo_dev
    print(f"\n{'='*50}\n[DEBUG - GPU CHECK]\nCUDA Available: {state['cuda_available']}\nGPU Name: {state['gpu_device_name']}\nYOLOv8 Device Target: {yolo_dev}\n{'='*50}\n", flush=True)
    
    
    print("[INFO] Loading TwinLiteNet Lane detector...", flush=True)
    twinlite_model = TwinLiteDetector()
    state["twinlite_device"] = str(twinlite_model.device)

    while True:
        has_frame = False
        with frame_lock:
            if raw_frame_buffer is not None:
                im0 = raw_frame_buffer.copy()
                raw_frame_buffer = None
                has_frame = True
                
        if not has_frame:
            time.sleep(0.01)
            continue
            
        # Ensure im0 is strictly 3-channel BGR
        if len(im0.shape) == 2:
            im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
        elif len(im0.shape) == 3 and im0.shape[2] == 4:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)

        im_infer = cv2.resize(im0, (INFER_WIDTH, INFER_HEIGHT), interpolation=cv2.INTER_LINEAR)

        fcw_triggered = False
        ldw_triggered = False

        if state["adas_enabled"]:
            # --- 1. YOLOv8 Forward Collision Warning ---
            # Predict objects without tracking overhead
            results = yolo_model.predict(im_infer, classes=[2, 5, 7], verbose=False, device=yolo_dev) # Cars, buses, trucks
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    cx = (x1 + x2) / 2
                    
                    # Draw box
                    cv2.rectangle(im_infer, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Ignore the ego-vehicle hood/dash which touches the bottom 20 pixels
                    if y2 > INFER_HEIGHT - 20:
                        continue
                        
                    # FCW Logic: If vehicle is in the center lane and box width is very large (close)
                    if CENTER_LANE_X_MIN < cx < CENTER_LANE_X_MAX:
                        if w > FCW_WARNING_WIDTH:
                            fcw_triggered = True
                            cv2.rectangle(im_infer, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                            cv2.putText(im_infer, "TOO CLOSE!", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- 2. TwinLiteNet Drivable Area & Lane Departure Warning ---
            da_mask, ll_mask = twinlite_model.detect(im_infer)
            
            if ll_mask is not None and da_mask is not None:
                h, w = ll_mask.shape
                car_center_x = w // 2
                
                if state.get("calibrate_requested"):
                    print("[INFO] Starting Lens Perspective Calibration...", flush=True)
                    state["calibrate_requested"] = False
                    state["calibration_frames_left"] = 30
                    state["calib_left_history"] = []
                    state["calib_right_history"] = []
                    
                if state["calibration_frames_left"] > 0:
                    l_x, r_x, v_y, _ = extract_lane_direct(ll_mask, car_center_x, None)
                    if l_x is not None and len(v_y) > 50: # Ensure we got a good, long highway read
                        # Normalize to full height array
                        full_l = np.full(h, np.nan)
                        full_r = np.full(h, np.nan)
                        full_l[v_y] = l_x
                        full_r[v_y] = r_x
                        state["calib_left_history"].append(full_l)
                        state["calib_right_history"].append(full_r)
                        
                    state["calibration_frames_left"] -= 1
                    
                    if state["calibration_frames_left"] == 0:
                        if len(state["calib_left_history"]) >= 15:
                            # We have enough frames! Calculate the perfect perspective taper
                            all_l = np.nanmedian(state["calib_left_history"], axis=0)
                            all_r = np.nanmedian(state["calib_right_history"], axis=0)
                            
                            valid_rows = ~np.isnan(all_l) & ~np.isnan(all_r)
                            y_vals = np.arange(h)[valid_rows]
                            widths = all_r[valid_rows] - all_l[valid_rows]
                            
                            # Fit a parabola to the width to perfectly capture barrel distortion taper
                            w_poly = np.polyfit(y_vals, widths, 2)
                            
                            calibrated_width = np.zeros(h, dtype=np.float32)
                            for y in range(h):
                                w = w_poly[0]*y**2 + w_poly[1]*y + w_poly[2]
                                calibrated_width[y] = max(0, w) # Width can't be negative
                                
                            # Find Vanishing Point Y (where width approaches 0)
                            vp_y = h
                            for y in range(h-1, -1, -1):
                                if calibrated_width[y] <= 5:
                                    vp_y = y
                                    break
                                    
                            # Find VP X (center at VP Y)
                            c_poly = np.polyfit(y_vals, (all_l[valid_rows] + all_r[valid_rows])/2.0, 2)
                            vp_x = c_poly[0]*vp_y**2 + c_poly[1]*vp_y + c_poly[2]
                            
                            # Save calibration
                            state["calibration"] = {
                                "calibrated_width": calibrated_width.tolist(),
                                "vp_x": float(vp_x),
                                "vp_y": int(vp_y)
                            }
                            os.makedirs('models', exist_ok=True)
                            with open('models/calibration.json', 'w') as f:
                                json.dump(state["calibration"], f)
                            print(f"[INFO] Lens Calibration Successful! VP: ({vp_x:.1f}, {vp_y})", flush=True)
                        else:
                            print("[ERROR] Lens Calibration Failed: Not enough clean lane lines.", flush=True)
                            
                # Normal Inference
                cw = None
                if state["calibration"] is not None:
                    cw = state["calibration"]["calibrated_width"]
                    
                raw_left_fitx, raw_right_fitx, valid_ploty, hood_top_y = extract_lane_direct(ll_mask, car_center_x, cw)
                
                # Apply temporal smoothing to the RAW tracks to completely eliminate frame-to-frame jitter
                raw_left_fitx = smooth_spline(raw_left_fitx, state["left_poly_history"])
                raw_right_fitx = smooth_spline(raw_right_fitx, state["right_poly_history"])
                
                if raw_left_fitx is not None and raw_right_fitx is not None:
                    
                    left_fitx = raw_left_fitx
                    right_fitx = raw_right_fitx
                    
                    # Compute vehicle deviation at the hood line (the closest visible point to the car)
                    lane_bottom_y = hood_top_y
                    left_bottom_x = left_fitx[0] # valid_ploty is descending
                    right_bottom_x = right_fitx[0]
                    
                    lane_width = right_bottom_x - left_bottom_x
                    lane_center_x = (left_bottom_x + right_bottom_x) / 2
                    drift = car_center_x - lane_center_x
                    
                    if lane_width > 0:
                        dist_left = car_center_x - left_bottom_x
                        dist_right = right_bottom_x - car_center_x
                        
                        ratio_left = dist_left / lane_width
                        ratio_right = dist_right / lane_width
                        
                        if ratio_left < 0.1: left_color, ldw_triggered = (0, 0, 255), True
                        elif ratio_left < 0.25: left_color = (0, 165, 255)
                        else: left_color = (0, 255, 0)
                            
                        if ratio_right < 0.1: right_color, ldw_triggered = (0, 0, 255), True
                        elif ratio_right < 0.25: right_color = (0, 165, 255)
                        else: right_color = (0, 255, 0)
                            
                        min_ratio = min(ratio_left, ratio_right)
                        if min_ratio < 0.1: fill_color = (0, 0, 255)
                        elif min_ratio < 0.25: fill_color = (0, 165, 255)
                        else: fill_color = (0, 255, 0)
                            
                        # Connect the top of the tracked lanes perfectly to the calibrated vanishing point
                        vp_x = w // 2
                        vp_y = 180
                        if state["calibration"] is not None:
                            vp_x = state["calibration"]["vp_x"]
                            vp_y = state["calibration"]["vp_y"]
                            
                        # Append the VP to the points!
                        left_pts = np.vstack([left_fitx, valid_ploty]).T
                        right_pts = np.vstack([right_fitx, valid_ploty]).T
                        
                        # Only extend if the neural net didn't already draw up to the horizon
                        if valid_ploty[-1] > vp_y + 10:
                            left_pts = np.vstack([left_pts, [vp_x, vp_y]])
                            right_pts = np.vstack([right_pts, [vp_x, vp_y]])
                        
                        pts_left = np.array([left_pts]).astype(np.int32)
                        pts_right = np.array([np.flipud(right_pts)]).astype(np.int32)
                        pts = np.hstack((pts_left, pts_right))
                        
                        overlay = np.zeros_like(im_infer)
                        cv2.fillPoly(overlay, [pts], fill_color)
                        
                        cv2.polylines(overlay, [pts_left[0]], False, left_color, thickness=6)
                        cv2.polylines(overlay, [pts_right[0]], False, right_color, thickness=6)
                        
                        alpha = 0.5 if ldw_triggered else 0.35
                        mask_indices = np.any(overlay != 0, axis=-1)
                        im_infer[mask_indices] = cv2.addWeighted(im_infer[mask_indices], 1.0 - alpha, overlay[mask_indices], alpha, 0)
                        
                        cv2.circle(im_infer, (int(lane_center_x), lane_bottom_y - 20), 8, (255, 255, 255), -1)
                        
                        if ldw_triggered:
                            direction = "RIGHT" if drift > 0 else "LEFT"
                            cv2.putText(im_infer, f"LANE DEPARTURE: {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # Update global state for UI alerts
        state["fcw_warning"] = fcw_triggered
        state["ldw_warning"] = ldw_triggered

        # Encode to JPEG for the web interface
        _, buf = cv2.imencode('.jpg', im_infer, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
            cv2.putText(wait_img, "Initializing ADAS models...",
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
        "adas_enabled": state["adas_enabled"],
        "fcw_warning": state["fcw_warning"],
        "ldw_warning": state["ldw_warning"],
        "recording_duration": duration,
        "error": state["error"],
        "cuda_available": state["cuda_available"],
        "gpu_device_name": state["gpu_device_name"],
        "yolo_device": state["yolo_device"],
        "twinlite_device": state["twinlite_device"]
    })

@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    data = request.json
    state["recording"] = data.get("enabled", not state["recording"])
    return jsonify({"success": True, "recording": state["recording"]})

@app.route('/api/toggle_adas', methods=['POST'])
def api_toggle_adas():
    global state
    data = request.json
    state["adas_enabled"] = data.get('enabled', False)
    return jsonify({"adas_enabled": state["adas_enabled"]})



@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    global state
    state["calibrate_requested"] = True
    return jsonify({"status": "calibrating"})

if __name__ == '__main__':
    print("=" * 50, flush=True)
    
    print("Starting Capture thread...", flush=True)
    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_capture.start()

    print("Starting Inference thread...", flush=True)
    t_inference = threading.Thread(target=inference_loop, daemon=True)
    t_inference.start()

    print("Starting Flask server on port 5001...", flush=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
