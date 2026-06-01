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
    "hood_detection_done": False
}

if os.path.exists('models/calibration.json'):
    try:
        with open('models/calibration.json', 'r') as f:
            state["calibration"] = json.load(f)
            print("[INFO] Successfully loaded Stable VP Calibration.", flush=True)
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

def detect_hood_line(frame):
    """
    Detect the top edge of the car hood using horizontal gradient analysis.
    The hood creates a strong, wide horizontal edge in the bottom portion of the frame.
    Returns the y-coordinate of the hood line, or None if not detected.
    """
    h, w = frame.shape[:2]
    
    # Only search the bottom 40% of the frame
    search_top = int(h * 0.6)
    bottom_region = frame[search_top:, :]
    
    # Convert to grayscale
    if len(bottom_region.shape) == 3:
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = bottom_region
    
    # Apply slight blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute horizontal edges using Sobel (vertical gradient)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.abs(sobel_y)
    
    # Sum gradient magnitude across each row (strong horizontal edges will have high sums)
    row_gradient_sum = np.sum(abs_sobel, axis=1)
    
    # Normalize by width so the threshold is resolution-independent
    row_gradient_avg = row_gradient_sum / w
    
    # Find the first strong horizontal edge from the top of the search region
    # The hood edge is typically the strongest horizontal feature
    threshold = np.max(row_gradient_avg) * 0.4
    
    # Look for a sustained strong edge (at least a few consecutive rows above threshold)
    candidates = np.where(row_gradient_avg > threshold)[0]
    
    if len(candidates) == 0:
        return None
    
    # The hood line is the first strong edge encountered
    hood_y_in_region = candidates[0]
    hood_y_absolute = search_top + hood_y_in_region
    
    return hood_y_absolute

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
            search_max = min(int(car_center_x), w//2 + 50)
            if np.max(histogram[:search_max]) > 0:
                leftx_current = np.argmax(histogram[:search_max])
                
        if rightx_current is None and np.max(histogram[int(car_center_x):]) > 0:
            search_min = max(int(car_center_x), w//2 - 50)
            if np.max(histogram[search_min:]) > 0:
                rightx_current = np.argmax(histogram[search_min:]) + search_min
        
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
                    if CENTER_LANE_X_MIN < cx < CENTER_LANE_X_MAX:
                        if w > FCW_WARNING_WIDTH:
                            fcw_triggered = True
                            cv2.rectangle(im_infer, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                            cv2.putText(im_infer, "TOO CLOSE!", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- Hood Detection (first 10 frames) ---
            if not state["hood_detection_done"]:
                detected_y = detect_hood_line(im_infer)
                if detected_y is not None:
                    state["hood_detection_frames"].append(detected_y)
                
                if len(state["hood_detection_frames"]) >= 10:
                    hood_y_final = int(np.median(state["hood_detection_frames"]))
                    state["hood_y_detected"] = hood_y_final
                    state["hood_detection_done"] = True
                    
                    # Persist to disk
                    os.makedirs('models', exist_ok=True)
                    with open('models/hood_line.json', 'w') as f:
                        json.dump({"hood_y": hood_y_final}, f)
                    print(f"[INFO] Hood line detected and saved: y={hood_y_final} (of {INFER_HEIGHT})", flush=True)

            # --- 2. TwinLiteNet Drivable Area & Lane Departure Warning ---
            da_mask, ll_mask = twinlite_model.detect(im_infer)
            
            # Mask out everything below the detected hood line
            if state["hood_y_detected"] is not None and da_mask is not None and ll_mask is not None:
                da_mask[state["hood_y_detected"]:, :] = 0
                ll_mask[state["hood_y_detected"]:, :] = 0
            
            if ll_mask is not None and da_mask is not None:
                h, w = ll_mask.shape
                car_center_x = w // 2
                
                if state.get("calibrate_requested"):
                    print("[INFO] Starting Stable VP Calibration...", flush=True)
                    state["calibrate_requested"] = False
                    state["calibration_frames_left"] = 30
                    state["calib_left_history"] = []
                    state["calib_right_history"] = []
                    
                l_x, l_y, r_x, r_y = extract_window_points(ll_mask, car_center_x)
                
                if state["calibration_frames_left"] > 0:
                    if len(l_x) >= 6:
                        # Top half of line is least distorted
                        half = len(l_x) // 2
                        p_l = np.polyfit(l_y[half:], l_x[half:], 1)
                        state["calib_left_history"].append(p_l)
                    if len(r_x) >= 6:
                        half = len(r_x) // 2
                        p_r = np.polyfit(r_y[half:], r_x[half:], 1)
                        state["calib_right_history"].append(p_r)
                        
                    state["calibration_frames_left"] -= 1
                    
                    if state["calibration_frames_left"] == 0:
                        if len(state["calib_left_history"]) >= 15 and len(state["calib_right_history"]) >= 15:
                            avg_pl = np.mean(state["calib_left_history"], axis=0)
                            avg_pr = np.mean(state["calib_right_history"], axis=0)
                            
                            m1, b1 = avg_pl
                            m2, b2 = avg_pr
                            
                            vp_y = (b2 - b1) / (m1 - m2)
                            vp_x = m1 * vp_y + b1
                            
                            state["calibration"] = {"vp_x": float(vp_x), "vp_y": int(vp_y)}
                            os.makedirs('models', exist_ok=True)
                            with open('models/calibration.json', 'w') as f:
                                json.dump(state["calibration"], f)
                            print(f"[INFO] Stable VP Calibration Successful! VP: ({vp_x:.1f}, {vp_y})", flush=True)
                        else:
                            print("[ERROR] Calibration Failed: Not enough clean lane lines.", flush=True)
                            
                # --- INFERENCE ---
                da_nonzero = da_mask.nonzero()
                if len(da_nonzero[0]) > 0:
                    horizon_y = int(np.min(da_nonzero[0]))
                    hood_y = int(np.max(da_nonzero[0]))
                else:
                    horizon_y = 180
                    hood_y = h - 20
                    
                # Clamp hood_y to the detected hood line
                if state["hood_y_detected"] is not None:
                    hood_y = min(hood_y, state["hood_y_detected"])
                    
                l_path = np.full(h, np.nan)
                if len(l_x) >= 2:
                    valid_y = np.arange(min(l_y), max(l_y) + 1)
                    l_path[valid_y] = np.interp(valid_y, l_y[::-1], l_x[::-1])
                    
                r_path = np.full(h, np.nan)
                if len(r_x) >= 2:
                    valid_y = np.arange(min(r_y), max(r_y) + 1)
                    r_path[valid_y] = np.interp(valid_y, r_y[::-1], r_x[::-1])
                    
                smoothed_l = smooth_path(l_path, state["left_poly_history"])
                smoothed_r = smooth_path(r_path, state["right_poly_history"])
                
                if smoothed_l is not None or smoothed_r is not None:
                    valid_l = ~np.isnan(smoothed_l) if smoothed_l is not None else np.zeros(h, dtype=bool)
                    if np.any(valid_l):
                        smoothed_l[valid_l] = gaussian_filter1d(smoothed_l[valid_l], sigma=5)
                        
                    valid_r = ~np.isnan(smoothed_r) if smoothed_r is not None else np.zeros(h, dtype=bool)
                    if np.any(valid_r):
                        smoothed_r[valid_r] = gaussian_filter1d(smoothed_r[valid_r], sigma=5)
                        
                    vp_x = w // 2
                    vp_y = 180
                    if state["calibration"] is not None:
                        vp_x = state["calibration"]["vp_x"]
                        vp_y = state["calibration"]["vp_y"]
                        
                    lane_bottom_y = hood_y
                    
                    def get_ideal_width(y):
                        # Pinhole perspective geometric taper
                        return 400 * (y - vp_y) / (h - vp_y)
                        
                    def project_path(path_arr, min_bound, max_bound):
                        valid_mask = ~np.isnan(path_arr)
                        if not np.any(valid_mask):
                            return np.array([]), np.array([])
                            
                        y_indices = np.where(valid_mask)[0]
                        min_y = max(np.min(y_indices), min_bound)
                        max_y = min(np.max(y_indices), max_bound)
                        
                        if min_y > max_y:
                            return np.array([]), np.array([])
                            
                        poly_y = np.arange(min_y, max_y + 1)
                        poly_x = path_arr[poly_y]
                        
                        # Top perspective projection to horizon
                        y0 = min_y
                        x0 = poly_x[0]
                        top_y = np.arange(min_bound, min_y)
                        if len(top_y) > 0:
                            if y0 != vp_y:
                                top_x = vp_x + (x0 - vp_x) * (top_y - vp_y) / (y0 - vp_y)
                            else:
                                top_x = np.full_like(top_y, x0)
                        else:
                            top_x = np.array([])
                            
                        # Bottom perspective projection to hood
                        y1 = max_y
                        x1 = poly_x[-1]
                        bot_y = np.arange(max_y + 1, max_bound + 1)
                        if len(bot_y) > 0:
                            if y1 != vp_y:
                                bot_x = vp_x + (x1 - vp_x) * (bot_y - vp_y) / (y1 - vp_y)
                            else:
                                bot_x = np.full_like(bot_y, x1)
                        else:
                            bot_x = np.array([])
                            
                        final_x = np.concatenate([top_x, poly_x, bot_x])
                        final_y = np.concatenate([top_y, poly_y, bot_y])
                        return final_x, final_y
                        
                    has_l = np.any(valid_l)
                    has_r = np.any(valid_r)
                    
                    if has_l and has_r:
                        left_fitx, ploty_l = project_path(smoothed_l, horizon_y, hood_y)
                        right_fitx, ploty_r = project_path(smoothed_r, horizon_y, hood_y)
                        
                    elif has_l:
                        left_fitx, ploty_l = project_path(smoothed_l, horizon_y, hood_y)
                        ploty_r = ploty_l
                        right_fitx = left_fitx + get_ideal_width(ploty_r)
                        
                    elif has_r:
                        right_fitx, ploty_r = project_path(smoothed_r, horizon_y, hood_y)
                        ploty_l = ploty_r
                        left_fitx = right_fitx - get_ideal_width(ploty_l)
                        
                    else:
                        left_fitx = np.array([])
                        right_fitx = np.array([])
                        
                    if len(left_fitx) > 0 and len(right_fitx) > 0:
                        left_pts = np.vstack([left_fitx, ploty_l]).T
                        right_pts = np.vstack([right_fitx, ploty_r]).T
                        
                        left_bottom_x = left_fitx[-1]
                        right_bottom_x = right_fitx[-1]
                        
                        lane_width = right_bottom_x - left_bottom_x
                        lane_center_x = (left_bottom_x + right_bottom_x) / 2
                        drift = car_center_x - lane_center_x
                        
                        if lane_width > 0:
                            dist_left = car_center_x - left_bottom_x
                            dist_right = right_bottom_x - car_center_x
                            
                            ratio_left = dist_left / lane_width
                            ratio_right = dist_right / lane_width
                            
                            ldw_triggered = False
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
