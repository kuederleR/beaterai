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
    "right_poly_history": [],
    "left_miny_history": [],
    "right_miny_history": [],
    "left_maxy_history": [],
    "right_maxy_history": [],
    "last_left_x": None,
    "last_right_x": None,
    "calibrate_requested": False,
    "calibration_frames_left": 0,
    "calib_left_history": [],
    "calib_right_history": [],
    "calibration": None,
    "hood_y_detected": None,
    "hood_detection_frames": [],
    "hood_detection_done": False,
    "car_center_x": 320,
    "calibrate_center_requested": False,
    "calibration_center_frames_left": 0,
    "calib_center_history": []
}

if os.path.exists('models/calibration.json'):
    try:
        with open('models/calibration.json', 'r') as f:
            calib_data = json.load(f)
            if "vp_x" in calib_data and "vp_y" in calib_data:
                state["calibration"] = {"vp_x": calib_data["vp_x"], "vp_y": calib_data["vp_y"]}
                print("[INFO] Successfully loaded Stable VP Calibration.", flush=True)
            if "car_center_x" in calib_data:
                state["car_center_x"] = calib_data["car_center_x"]
                print(f"[INFO] Loaded calibrated car center: x={state['car_center_x']}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load calibration matrix: {e}", flush=True)

if os.path.exists('models/hood_line.json'):
    try:
        with open('models/hood_line.json', 'r') as f:
            hood_data = json.load(f)
            state["hood_y_detected"] = hood_data["hood_y"]
            state["hood_detection_done"] = True
            print(f"[INFO] Loaded hood line from cache: y={state['hood_y_detected']}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load hood line: {e}", flush=True)

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
import warnings

def smooth_path(path, history, max_history=10):
    if path is None:
        return None
    history.append(path)
    if len(history) > max_history:
        history.pop(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(history, axis=0)

def smooth_scalar(val, history, max_history=5):
    if val is None:
        return None
    history.append(val)
    if len(history) > max_history:
        history.pop(0)
    return np.mean(history)

def detect_hood_from_frame(frame):
    """
    Detect the hood boundary by color-matching from the bottom of the frame upward.
    
    The bottom rows are always hood/dash. For each column in the center strip,
    we scan upward until the color clearly differs from the bottom reference.
    This finds the hood's upper edge — faint reflections above the hood are
    never reached because we stop at the first strong color transition.
    
    Returns the y-coordinate of the hood line, or None if detection fails.
    """
    h, w = frame.shape[:2]
    
    # Work in grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = frame.astype(np.float32)
    
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # The bottom 8 rows are guaranteed hood — use as per-column color reference
    hood_ref = np.mean(gray[h-8:h, :], axis=0)  # shape [w]
    
    # Adaptive threshold: must exceed hood's own variance by a clear margin
    hood_std = np.mean(np.std(gray[h-8:h, :], axis=0))
    threshold = max(hood_std * 4, 25)
    
    # Per-pixel difference from that column's hood reference
    diff = np.abs(gray - hood_ref[np.newaxis, :])
    
    # Only scan the center 50% of frame width (avoid side mirrors, edges)
    center_l = w // 4
    center_r = 3 * w // 4
    
    boundaries = []
    for x in range(center_l, center_r, 2):  # Every other column for speed
        consecutive = 0
        for y in range(h - 9, int(h * 0.3), -1):
            if diff[y, x] > threshold:
                consecutive += 1
                if consecutive >= 3:  # Need 3+ consecutive non-hood pixels
                    boundaries.append(y + consecutive)
                    break
            else:
                consecutive = 0
    
    if len(boundaries) < 20:
        return None
    
    hood_y = int(np.median(boundaries))
    
    # Sanity: hood must be in the bottom half of the frame
    if hood_y < h // 2:
        return None
    
    return hood_y

def extract_window_points(ll_mask, car_center_x, prev_l_x=None, prev_r_x=None):
    h, w = ll_mask.shape
    
    nwindows = 15
    margin = 40
    minpix = 15
    
    nonzero = ll_mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    if len(nonzeroy) == 0:
        return [], [], [], []
        
    start_y = np.max(nonzeroy)
    search_height = start_y - 150
    if search_height < 50:
        return [], [], [], []
        
    window_height = int(search_height / nwindows)
    
    leftx_current = prev_l_x
    rightx_current = prev_r_x
    
    if leftx_current is None or rightx_current is None:
        bottom_band = ll_mask[start_y - 40:start_y + 1, :]
        histogram = np.sum(bottom_band, axis=0)
        
        if leftx_current is None and np.max(histogram[:int(car_center_x)]) > 0:
            leftx_current = np.argmax(histogram[:int(car_center_x)])
                
        if rightx_current is None and np.max(histogram[int(car_center_x):]) > 0:
            rightx_current = np.argmax(histogram[int(car_center_x):]) + int(car_center_x)
        
    left_x_pts = []
    left_y_pts = []
    right_x_pts = []
    right_y_pts = []
    
    for window in range(nwindows):
        win_y_low = start_y - (window + 1) * window_height
        win_y_high = start_y - window * window_height
        win_y_center = (win_y_low + win_y_high) // 2
        
        if leftx_current is not None:
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
                left_x_pts.append(leftx_current)
                left_y_pts.append(win_y_center)
            else:
                leftx_current = None # Stop tracking to prevent jumping
                
        if rightx_current is not None:
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))
                right_x_pts.append(rightx_current)
                right_y_pts.append(win_y_center)
            else:
                rightx_current = None
                
    return left_x_pts, left_y_pts, right_x_pts, right_y_pts

def find_lane_boundaries_radial(da_mask, ll_mask):
    """
    Find left and right lane line boundaries by tracing rays radiating outward
    from the bottom center of the drivable area.
    """
    h, w = da_mask.shape
    
    # Start point (cx, cy): bottom center of drivable area
    da_nonzero = da_mask.nonzero()
    if len(da_nonzero[0]) == 0:
        return [], []
        
    cy = int(np.max(da_nonzero[0])) # Bottom of drivable area
    # Center of the drivable area at the bottom rows
    bottom_y_min = max(0, cy - 20)
    bottom_da_pts = da_mask[bottom_y_min:cy + 1, :].nonzero()
    if len(bottom_da_pts[1]) > 0:
        cx = int(np.median(bottom_da_pts[1]))
    else:
        cx = w // 2
        
    left_points = []
    right_points = []
    
    max_dist = int(np.sqrt(h**2 + w**2))
    
    # Trace left rays (95 to 195 degrees, every 3 degrees)
    for angle_deg in range(95, 195, 3):
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for r in range(5, max_dist, 3):
            px = int(cx + r * cos_t)
            py = int(cy - r * sin_t) # y-axis goes down in image coordinates
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
                
            # Stop if we exit the drivable area
            if da_mask[py, px] == 0:
                break
                
            # If we hit a lane line pixel, record and stop
            if ll_mask[py, px] > 0:
                left_points.append((px, py))
                break
                
    # Trace right rays (85 to -15 degrees, every 3 degrees)
    for angle_deg in range(85, -15, -3):
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for r in range(5, max_dist, 3):
            px = int(cx + r * cos_t)
            py = int(cy - r * sin_t)
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
                
            # Stop if we exit the drivable area
            if da_mask[py, px] == 0:
                break
                
            # If we hit a lane line pixel, record and stop
            if ll_mask[py, px] > 0:
                right_points.append((px, py))
                break
                
    return left_points, right_points

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
                    
                    # Ignore the ego-vehicle hood/dash
                    yolo_hood_y = state["hood_y_detected"] if state["hood_y_detected"] is not None else INFER_HEIGHT - 20
                    if y2 > yolo_hood_y:
                        continue
                        
                    # FCW Logic: If vehicle is in the center lane and box width is very large (close)
                    current_center_x = state.get("car_center_x", INFER_WIDTH // 2)
                    center_lane_min = current_center_x - (INFER_WIDTH // 6)
                    center_lane_max = current_center_x + (INFER_WIDTH // 6)
                    if center_lane_min < cx < center_lane_max:
                        if w > FCW_WARNING_WIDTH:
                            fcw_triggered = True
                            cv2.rectangle(im_infer, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                            cv2.putText(im_infer, "TOO CLOSE!", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- Hood Detection (first 15 frames, bottom-up color scan) ---
            if not state["hood_detection_done"]:
                detected_y = detect_hood_from_frame(im_infer)
                if detected_y is not None:
                    state["hood_detection_frames"].append(detected_y)
                
                if len(state["hood_detection_frames"]) >= 15:
                    hood_y_final = int(np.median(state["hood_detection_frames"]))
                    state["hood_y_detected"] = hood_y_final
                    state["hood_detection_done"] = True
                    
                    # Persist to disk
                    os.makedirs('models', exist_ok=True)
                    with open('models/hood_line.json', 'w') as f:
                        json.dump({"hood_y": hood_y_final}, f)
                    print(f"[INFO] Hood line detected (color scan) and saved: y={hood_y_final} (of {INFER_HEIGHT})", flush=True)

            # --- 2. TwinLiteNet Drivable Area & Lane Departure Warning ---
            da_mask, ll_mask = twinlite_model.detect(im_infer)
            
            # Mask out everything below the detected hood line
            if state["hood_y_detected"] is not None and da_mask is not None and ll_mask is not None:
                da_mask[state["hood_y_detected"]:, :] = 0
                ll_mask[state["hood_y_detected"]:, :] = 0
            
            if ll_mask is not None and da_mask is not None:
                # 1. Plot drivable area (translucent green overlay)
                da_indices = da_mask > 0
                if np.any(da_indices):
                    overlay = np.zeros_like(im_infer)
                    overlay[da_indices] = (0, 255, 0)
                    alpha = 0.35
                    im_infer[da_indices] = cv2.addWeighted(im_infer[da_indices], 1.0 - alpha, overlay[da_indices], alpha, 0)
                
                # 2. Plot lane lines (solid yellow) and compute smooth lane boundaries
                ll_indices = ll_mask > 0
                if np.any(ll_indices):
                    im_infer[ll_indices] = (0, 255, 255)

                # Attempt to compute lane boundary polynomials using radial tracing
                left_pts, right_pts = find_lane_boundaries_radial(da_mask, ll_mask)
                left_poly = None
                right_poly = None

                # Fit quadratic x = f(y) if we have enough points
                if len(left_pts) >= 5:
                    lx = np.array([p[0] for p in left_pts])
                    ly = np.array([p[1] for p in left_pts])
                    try:
                        left_poly = np.polyfit(ly, lx, 2)
                    except Exception:
                        left_poly = None

                if len(right_pts) >= 5:
                    rx = np.array([p[0] for p in right_pts])
                    ry = np.array([p[1] for p in right_pts])
                    try:
                        right_poly = np.polyfit(ry, rx, 2)
                    except Exception:
                        right_poly = None

                # Fallback: use sliding window point extraction if polynomials missing
                if left_poly is None or right_poly is None:
                    prev_lx = state.get('last_left_x')
                    prev_rx = state.get('last_right_x')
                    l_x_pts, l_y_pts, r_x_pts, r_y_pts = extract_window_points(
                        ll_mask, state.get('car_center_x', INFER_WIDTH // 2), prev_lx, prev_rx)
                    if left_poly is None and len(l_x_pts) >= 4:
                        try:
                            left_poly = np.polyfit(np.array(l_y_pts), np.array(l_x_pts), 2)
                        except Exception:
                            left_poly = None
                    if right_poly is None and len(r_x_pts) >= 4:
                        try:
                            right_poly = np.polyfit(np.array(r_y_pts), np.array(r_x_pts), 2)
                        except Exception:
                            right_poly = None

                # Smooth polynomials using history buffers
                if left_poly is not None:
                    left_poly_sm = smooth_path(left_poly, state['left_poly_history'], max_history=5)
                else:
                    left_poly_sm = None
                if right_poly is not None:
                    right_poly_sm = smooth_path(right_poly, state['right_poly_history'], max_history=5)
                else:
                    right_poly_sm = None

                # Draw smoothed lane boundary polylines (blue)
                h_im = im_infer.shape[0]
                ys = np.linspace(int(h_im * 0.4), h_im - 1, num=100).astype(np.int32)
                if left_poly_sm is not None:
                    xs = np.polyval(left_poly_sm, ys)
                    pts = np.stack([xs, ys], axis=1).astype(np.int32)
                    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < im_infer.shape[1])]
                    if len(pts) > 1:
                        cv2.polylines(im_infer, [pts.reshape(-1, 1, 2)], isClosed=False, color=(255, 0, 0), thickness=3)
                        state['last_left_x'] = int(np.mean(pts[-5:, 0])) if len(pts) >= 5 else int(pts[-1, 0])
                if right_poly_sm is not None:
                    xs = np.polyval(right_poly_sm, ys)
                    pts = np.stack([xs, ys], axis=1).astype(np.int32)
                    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < im_infer.shape[1])]
                    if len(pts) > 1:
                        cv2.polylines(im_infer, [pts.reshape(-1, 1, 2)], isClosed=False, color=(255, 0, 0), thickness=3)
                        state['last_right_x'] = int(np.mean(pts[-5:, 0])) if len(pts) >= 5 else int(pts[-1, 0])

                ldw_triggered = False

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
        "twinlite_device": state["twinlite_device"],
        "car_center_x": state["car_center_x"],
        "calibrating_center": state["calibration_center_frames_left"] > 0,
        "calibration_center_progress": 150 - state["calibration_center_frames_left"]
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

@app.route('/api/calibrate_center', methods=['POST'])
def api_calibrate_center():
    global state
    state["calibrate_center_requested"] = True
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
