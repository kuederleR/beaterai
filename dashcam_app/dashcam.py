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
    "calibrate_requested": False,
    "calibration": None,
    "calibration_buffer": [],
    "calibration_frames_left": 0
}

if os.path.exists('models/calibration.json'):
    try:
        with open('models/calibration.json', 'r') as f:
            state["calibration"] = json.load(f)
            state["calibration"]["M"] = np.array(state["calibration"]["M"], dtype=np.float32)
            state["calibration"]["Minv"] = np.array(state["calibration"]["Minv"], dtype=np.float32)
            print("[INFO] Successfully loaded IPM Calibration Matrix.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load calibration matrix: {e}", flush=True)
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

def extract_raw_vp(ll_mask):
    h, w = ll_mask.shape
    bottom_band = ll_mask[h - 40:h, :]
    histogram = np.sum(bottom_band, axis=0)
    
    center_x = w // 2
    left_search_min, left_search_max = max(0, center_x - 300), max(0, center_x - 20)
    right_search_min, right_search_max = min(w, center_x + 20), min(w, center_x + 300)
    
    leftx_base = None
    if left_search_max > left_search_min and np.max(histogram[left_search_min:left_search_max]) > 5:
        leftx_base = np.argmax(histogram[left_search_min:left_search_max]) + left_search_min
        
    rightx_base = None
    if right_search_max > right_search_min and np.max(histogram[right_search_min:right_search_max]) > 5:
        rightx_base = np.argmax(histogram[right_search_min:right_search_max]) + right_search_min
        
    if leftx_base is None or rightx_base is None:
        return None
        
    nwindows = 10
    window_height = int((h*0.4) / nwindows)
    margin = 40
    minpix = 15
    
    left_x_pts, left_y_pts = [], []
    right_x_pts, right_y_pts = [], []
    
    leftx_current, rightx_current = leftx_base, rightx_base
    y_indices, x_indices = np.nonzero(ll_mask)
    
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_y_center = (win_y_low + win_y_high) // 2
        
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        good_left = ((y_indices >= win_y_low) & (y_indices < win_y_high) & 
                     (x_indices >= win_xleft_low) & (x_indices < win_xleft_high)).nonzero()[0]
        if len(good_left) > minpix:
            leftx_current = int(np.mean(x_indices[good_left]))
            left_x_pts.append(leftx_current)
            left_y_pts.append(win_y_center)
            
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        good_right = ((y_indices >= win_y_low) & (y_indices < win_y_high) & 
                      (x_indices >= win_xright_low) & (x_indices < win_xright_high)).nonzero()[0]
        if len(good_right) > minpix:
            rightx_current = int(np.mean(x_indices[good_right]))
            right_x_pts.append(rightx_current)
            right_y_pts.append(win_y_center)
            
    if len(left_x_pts) < 5 or len(right_x_pts) < 5:
        return None
        
    p_left = np.polyfit(left_y_pts, left_x_pts, 1)
    p_right = np.polyfit(right_y_pts, right_x_pts, 1)
    
    m1, b1 = p_left
    m2, b2 = p_right
    if abs(m1 - m2) < 1e-5: return None
    
    vp_y = (b2 - b1) / (m1 - m2)
    vp_x = m1 * vp_y + b1
    
    if vp_y > h or vp_y < 0: return None
    
    return {"vp_x": vp_x, "vp_y": vp_y, "m1": m1, "b1": b1, "m2": m2, "b2": b2}

def extract_lane_bev(ll_mask, M):
    h, w = ll_mask.shape
    bev_mask = cv2.warpPerspective(ll_mask, M, (w, h), flags=cv2.INTER_LINEAR)
    
    histogram = np.sum(bev_mask[h//2:, :], axis=0)
    center_x = w // 2
    
    leftx_base = np.argmax(histogram[:center_x]) if np.max(histogram[:center_x]) > 5 else None
    rightx_base = np.argmax(histogram[center_x:]) + center_x if np.max(histogram[center_x:]) > 5 else None
    
    nwindows = 15
    window_height = h // nwindows
    margin = 40
    minpix = 15
    
    left_x_pts, left_y_pts = [], []
    right_x_pts, right_y_pts = [], []
    
    leftx_current, rightx_current = leftx_base, rightx_base
    y_indices, x_indices = np.nonzero(bev_mask)
    
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_y_center = (win_y_low + win_y_high) // 2
        
        if leftx_current is not None:
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            good_left = ((y_indices >= win_y_low) & (y_indices < win_y_high) & 
                         (x_indices >= win_xleft_low) & (x_indices < win_xleft_high)).nonzero()[0]
            if len(good_left) > minpix:
                leftx_current = int(np.mean(x_indices[good_left]))
            left_x_pts.append(leftx_current)
            left_y_pts.append(win_y_center)
            
        if rightx_current is not None:
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            good_right = ((y_indices >= win_y_low) & (y_indices < win_y_high) & 
                          (x_indices >= win_xright_low) & (x_indices < win_xright_high)).nonzero()[0]
            if len(good_right) > minpix:
                rightx_current = int(np.mean(x_indices[good_right]))
            right_x_pts.append(rightx_current)
            right_y_pts.append(win_y_center)
            
    ploty = np.linspace(0, h-1, num=h)
    left_fitx, right_fitx = None, None
    
    if len(left_x_pts) == nwindows:
        ly = np.array(left_y_pts)[::-1]
        lx = np.array(left_x_pts)[::-1]
        raw_fitx = np.interp(ploty, ly, lx)
        left_fitx = gaussian_filter1d(raw_fitx, sigma=15.0)
        
    if len(right_x_pts) == nwindows:
        ry = np.array(right_y_pts)[::-1]
        rx = np.array(right_x_pts)[::-1]
        raw_fitx = np.interp(ploty, ry, rx)
        right_fitx = gaussian_filter1d(raw_fitx, sigma=15.0)
        
    return left_fitx, right_fitx, ploty

def extract_lane_splines(ll_mask, da_mask, center_x):
    """Uses a sliding window to isolate the ego lane and fit Gaussian splines, preventing perspective banana-ing."""
    h, w = ll_mask.shape
    
    # 1. Automatically detect the top of the car hood using the Drivable Area (da_mask)
    # We analyze the center 1/3rd of the image from the bottom up to find where the road actually begins.
    center_da = da_mask[:, int(w*0.33):int(w*0.66)]
    road_pixels_per_row = np.sum(center_da, axis=1)
    
    hood_top_y = h - 1
    for y in range(h-1, int(h*0.5), -1):
        # If at least 10% of the center width is classified as drivable road, we have cleared the hood.
        if road_pixels_per_row[y] > (w * 0.33 * 0.1): 
            hood_top_y = y
            break
            
    # Add a safety margin (e.g., 20 pixels) above the hood to avoid any distorted/noisy edge pixels
    hood_top_y = max(int(h * 0.5), hood_top_y - 20)
    
    # Our clean road search space is strictly between the horizon (h*0.5) and the hood
    search_top = int(h * 0.5)
    search_bottom = hood_top_y
    search_height = search_bottom - search_top
    
    if search_height < 50: # If we can't see enough road, abort
        return None, None, hood_top_y
    
    # 2. Extract starting base using a localized histogram of ONLY the very bottom of the road
    # By strictly scanning the bottom 40 pixels, we avoid pulling the line inwards ("bending in").
    bottom_band = ll_mask[max(search_top, search_bottom - 40) : search_bottom, :]
    histogram = np.sum(bottom_band, axis=0)
    
    # Ego-Band Searching: We physically restrict the base search to where the ego lane lines 
    # MUST logically exist relative to the camera center (e.g. 20 to 250 pixels away).
    # This completely blinds the algorithm to the outer boundaries of the highway.
    left_search_min = max(0, center_x - 250)
    left_search_max = max(0, center_x - 20)
    
    if left_search_max > left_search_min and np.max(histogram[left_search_min:left_search_max]) > 10:
        leftx_base = np.argmax(histogram[left_search_min:left_search_max]) + left_search_min
    else:
        leftx_base = None
        
    right_search_min = min(w, center_x + 20)
    right_search_max = min(w, center_x + 250)
    
    if right_search_max > right_search_min and np.max(histogram[right_search_min:right_search_max]) > 10:
        rightx_base = np.argmax(histogram[right_search_min:right_search_max]) + right_search_min
    else:
        rightx_base = None
    
    nwindows = 15
    window_height = int(search_height / nwindows)
    margin = 40 # Increased slightly to track curved dashed lines reliably
    minpix = 15
    
    left_x_pts, left_y_pts = [], []
    right_x_pts, right_y_pts = [], []
    
    leftx_current, rightx_current = leftx_base, rightx_base
    
    y_indices, x_indices = np.nonzero(ll_mask)
    
    for window in range(nwindows):
        win_y_low = search_bottom - (window + 1) * window_height
        win_y_high = search_bottom - window * window_height
        win_y_center = (win_y_low + win_y_high) // 2
        
        if leftx_current is not None:
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            good_left_inds = ((y_indices >= win_y_low) & (y_indices < win_y_high) & 
                              (x_indices >= win_xleft_low) & (x_indices < win_xleft_high)).nonzero()[0]
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(x_indices[good_left_inds]))
            left_x_pts.append(leftx_current)
            left_y_pts.append(win_y_center)
                
        if rightx_current is not None:
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            good_right_inds = ((y_indices >= win_y_low) & (y_indices < win_y_high) & 
                               (x_indices >= win_xright_low) & (x_indices < win_xright_high)).nonzero()[0]
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(x_indices[good_right_inds]))
            right_x_pts.append(rightx_current)
            right_y_pts.append(win_y_center)
                
    # 3. Robust Gaussian Spline Generation
    # We replace rigid parabolic equations with fluid vector splines that perfectly match perspective view curves.
    ploty = np.linspace(int(h*0.5), hood_top_y, num=h//2)
    left_fitx, right_fitx = None, None
    
    if len(left_x_pts) == nwindows:
        # Interpolate requires strictly increasing x-axis (our y coordinates in image space)
        ly = np.array(left_y_pts)[::-1]
        lx = np.array(left_x_pts)[::-1]
        raw_fitx = np.interp(ploty, ly, lx)
        left_fitx = gaussian_filter1d(raw_fitx, sigma=8.0)
        
    if len(right_x_pts) == nwindows:
        ry = np.array(right_y_pts)[::-1]
        rx = np.array(right_x_pts)[::-1]
        raw_fitx = np.interp(ploty, ry, rx)
        right_fitx = gaussian_filter1d(raw_fitx, sigma=8.0)
        
    return left_fitx, right_fitx, ploty, hood_top_y

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
                    print("[INFO] Starting robust 30-frame calibration...", flush=True)
                    state["calibrate_requested"] = False
                    state["calibration_frames_left"] = 30
                    state["calibration_buffer"] = []
                    
                if state["calibration_frames_left"] > 0:
                    raw_vp = extract_raw_vp(ll_mask)
                    if raw_vp is not None:
                        state["calibration_buffer"].append(raw_vp)
                    state["calibration_frames_left"] -= 1
                    
                    if state["calibration_frames_left"] == 0:
                        if len(state["calibration_buffer"]) >= 15:
                            # We have enough good frames! Calculate rolling median.
                            med_vp_x = np.median([c["vp_x"] for c in state["calibration_buffer"]])
                            med_vp_y = np.median([c["vp_y"] for c in state["calibration_buffer"]])
                            med_m1 = np.median([c["m1"] for c in state["calibration_buffer"]])
                            med_b1 = np.median([c["b1"] for c in state["calibration_buffer"]])
                            med_m2 = np.median([c["m2"] for c in state["calibration_buffer"]])
                            med_b2 = np.median([c["b2"] for c in state["calibration_buffer"]])
                            
                            # Gently enforce symmetry towards camera optical center
                            target_center_x = w / 2
                            med_vp_x = med_vp_x * 0.5 + target_center_x * 0.5
                            
                            # Calculate extended horizon for deeper draw distance (shrink into horizon)
                            bottom_y = h
                            top_y = med_vp_y + 15
                            
                            src = np.float32([
                                [med_m1 * bottom_y + med_b1, bottom_y],
                                [med_m2 * bottom_y + med_b2, bottom_y],
                                [med_m2 * top_y + med_b2, top_y],
                                [med_m1 * top_y + med_b1, top_y]
                            ])
                            
                            margin_x = w * 0.25
                            dst = np.float32([
                                [margin_x, h],
                                [w - margin_x, h],
                                [w - margin_x, 0],
                                [margin_x, 0]
                            ])
                            
                            M = cv2.getPerspectiveTransform(src, dst)
                            Minv = cv2.getPerspectiveTransform(dst, src)
                            
                            state["calibration"] = {"M": M, "Minv": Minv, "top_y": int(top_y)}
                            
                            os.makedirs('models', exist_ok=True)
                            with open('models/calibration.json', 'w') as f:
                                json.dump({
                                    "M": M.tolist(), 
                                    "Minv": Minv.tolist(),
                                    "top_y": int(top_y)
                                }, f)
                            print("[INFO] Robust IPM Calibration Successful and Saved!", flush=True)
                        else:
                            print("[ERROR] Too much noise during calibration phase. Try again on a straighter road.", flush=True)
                    
                if state["calibration"] is not None:
                    # --- BIRD'S EYE VIEW PERCEPTION PIPELINE ---
                    M = state["calibration"]["M"]
                    Minv = state["calibration"]["Minv"]
                    
                    raw_left_fitx, raw_right_fitx, ploty = extract_lane_bev(ll_mask, M)
                    
                    raw_left_fitx = smooth_spline(raw_left_fitx, state["left_poly_history"])
                    raw_right_fitx = smooth_spline(raw_right_fitx, state["right_poly_history"])
                    
                    if raw_left_fitx is not None or raw_right_fitx is not None:
                        ideal_lane_width = w * 0.45 # Constant physical width in BEV
                        
                        if raw_left_fitx is not None and raw_right_fitx is not None:
                            center_fitx = (raw_left_fitx + raw_right_fitx) / 2.0
                        elif raw_left_fitx is not None:
                            center_fitx = raw_left_fitx + ideal_lane_width / 2.0
                        else:
                            center_fitx = raw_right_fitx - ideal_lane_width / 2.0
                            
                        left_fitx = center_fitx - ideal_lane_width / 2.0
                        right_fitx = center_fitx + ideal_lane_width / 2.0
                        
                        lane_width = ideal_lane_width
                        lane_center_x = center_fitx[-1]
                        drift = car_center_x - lane_center_x
                        
                        dist_left = car_center_x - left_fitx[-1]
                        dist_right = right_fitx[-1] - car_center_x
                        
                        ratio_left = dist_left / lane_width
                        ratio_right = dist_right / lane_width
                        
                        # Dynamic color grading
                        if ratio_left < 0.1:
                            left_color, ldw_triggered = (0, 0, 255), True
                        elif ratio_left < 0.25: left_color = (0, 165, 255)
                        else: left_color = (0, 255, 0)
                            
                        if ratio_right < 0.1:
                            right_color, ldw_triggered = (0, 0, 255), True
                        elif ratio_right < 0.25: right_color = (0, 165, 255)
                        else: right_color = (0, 255, 0)
                            
                        fill_color = (0, 0, 255) if ldw_triggered else (0, 255, 0)
                        alpha = 0.5 if ldw_triggered else 0.35
                        
                        # Draw in BEV Canvas
                        bev_canvas = np.zeros_like(im_infer)
                        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                        pts = np.hstack((pts_left, pts_right))
                        
                        cv2.fillPoly(bev_canvas, np.int_([pts]), fill_color)
                        cv2.polylines(bev_canvas, np.int_([pts_left]), False, left_color, thickness=10)
                        cv2.polylines(bev_canvas, np.int_([pts_right]), False, right_color, thickness=10)
                        
                        # Warp back to perspective perfectly
                        perspective_overlay = cv2.warpPerspective(bev_canvas, Minv, (w, h))
                        
                        mask_indices = np.any(perspective_overlay != 0, axis=-1)
                        im_infer[mask_indices] = cv2.addWeighted(im_infer[mask_indices], 1.0 - alpha, perspective_overlay[mask_indices], alpha, 0)
                        
                        if ldw_triggered:
                            direction = "RIGHT" if drift > 0 else "LEFT"
                            cv2.putText(im_infer, f"LANE DEPARTURE: {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                else:
                    # --- FALLBACK PERSPECTIVE PIPELINE ---
                    raw_left_fitx, raw_right_fitx, ploty, hood_top_y = extract_lane_splines(ll_mask, da_mask, car_center_x)
                    
                    # Apply temporal smoothing to the RAW tracks to completely eliminate frame-to-frame jitter
                    raw_left_fitx = smooth_spline(raw_left_fitx, state["left_poly_history"])
                    raw_right_fitx = smooth_spline(raw_right_fitx, state["right_poly_history"])
                    
                    if raw_left_fitx is not None or raw_right_fitx is not None:
                        # --- UNIFIED PREDICTED PATH ALGORITHM ---
                        # Default perspective parameters for ghosting (wide at hood, narrow at horizon)
                        ideal_lane_width = np.linspace(20, 350, num=len(ploty))
                        
                        if raw_left_fitx is not None and raw_right_fitx is not None:
                            # We have both lines! Calculate the true center driving path.
                            center_fitx = (raw_left_fitx + raw_right_fitx) / 2.0
                            
                            # Dynamically calculate the perspective taper from the actual tracked lines
                            bottom_width = raw_right_fitx[-1] - raw_left_fitx[-1]
                            top_width = raw_right_fitx[0] - raw_left_fitx[0]
                            # Ensure realistic bounds just in case of noise
                            bottom_width = max(100, min(500, bottom_width))
                            top_width = max(5, min(100, top_width))
                            
                            ideal_lane_width = np.linspace(top_width, bottom_width, num=len(ploty))
                            
                        elif raw_left_fitx is not None:
                            # Left line only! Infer center and "ghost" the right line.
                            center_fitx = raw_left_fitx + ideal_lane_width / 2.0
                        else:
                            # Right line only! Infer center and "ghost" the left line.
                            center_fitx = raw_right_fitx - ideal_lane_width / 2.0
                            
                        # Project the perfect, mathematically symmetrical predicted path!
                        left_fitx = center_fitx - ideal_lane_width / 2.0
                        right_fitx = center_fitx + ideal_lane_width / 2.0
                        
                        # Compute vehicle deviation at the hood line (the closest visible point to the car)
                        lane_bottom_y = hood_top_y
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
                            
                            # Dynamic color grading logic (BGR format)
                            # Left Line
                            if ratio_left < 0.1:
                                left_color = (0, 0, 255) # Red
                                ldw_triggered = True
                            elif ratio_left < 0.25:
                                left_color = (0, 165, 255) # Orange
                            else:
                                left_color = (0, 255, 0) # Green
                                
                            # Right Line
                            if ratio_right < 0.1:
                                right_color = (0, 0, 255) # Red
                                ldw_triggered = True
                            elif ratio_right < 0.25:
                                right_color = (0, 165, 255) # Orange
                            else:
                                right_color = (0, 255, 0) # Green
                                
                            # Polygon fill matches the most dangerous side
                            min_ratio = min(ratio_left, ratio_right)
                            if min_ratio < 0.1:
                                fill_color = (0, 0, 255)
                                ldw_triggered = True
                            elif min_ratio < 0.25:
                                fill_color = (0, 165, 255)
                            else:
                                fill_color = (0, 255, 0)
                                
                            # Build the Ego Lane polygon
                            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int32)
                            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]).astype(np.int32)
                            pts = np.hstack((pts_left, pts_right))
                            
                            overlay = np.zeros_like(im_infer)
                            cv2.fillPoly(overlay, [pts], fill_color)
                            
                            # Draw the thick vectorized lane lines
                            cv2.polylines(overlay, [pts_left[0]], False, left_color, thickness=6)
                            cv2.polylines(overlay, [pts_right[0]], False, right_color, thickness=6)
                            
                            # Alpha blend ONLY the filled region for maximum speed
                            alpha = 0.5 if ldw_triggered else 0.35
                            mask_indices = np.any(overlay != 0, axis=-1)
                            im_infer[mask_indices] = cv2.addWeighted(im_infer[mask_indices], 1.0 - alpha, overlay[mask_indices], alpha, 0)
                            
                            # Draw lane center tracking dot
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
