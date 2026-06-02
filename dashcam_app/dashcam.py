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
                    current_center_x = state.get("car_center_x", INFER_WIDTH // 2)
                    center_lane_min = current_center_x - (INFER_WIDTH // 6)
                    center_lane_max = current_center_x + (INFER_WIDTH // 6)
                    if center_lane_min < cx < center_lane_max:
                        if w > FCW_WARNING_WIDTH:
                            fcw_triggered = True
                            cv2.rectangle(im_infer, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                            cv2.putText(im_infer, "TOO CLOSE!", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- 2. TwinLiteNet Drivable Area & Lane Departure Warning ---
            da_mask, ll_mask = twinlite_model.detect(im_infer)
            
            if ll_mask is not None and da_mask is not None:
                # New approach: Segment drivable area into lanes using lane lines as dividers
                car_center_x = state.get("car_center_x", INFER_WIDTH // 2)
                h_im, w_im = im_infer.shape[:2]
                
                # Step 1: Intelligently extend lane lines through full drivable area height
                # Find vertical extent of drivable area
                da_rows = np.any(da_mask > 0, axis=1)
                da_top = np.argmax(da_rows) if np.any(da_rows) else 0
                da_bottom = len(da_rows) - 1 - np.argmax(da_rows[::-1]) if np.any(da_rows) else h_im - 1
                
                # First, connect existing lane line fragments vertically
                kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
                ll_connected = cv2.morphologyEx(ll_mask, cv2.MORPH_CLOSE, kernel_vertical)
                
                # Create extended lane line mask - only extend strong vertical structures
                ll_extended = ll_connected.copy()
                
                # For each column, check if it has a strong vertical lane line structure
                for x in range(w_im):
                    col_ll = ll_connected[:, x]
                    col_da = da_mask[:, x]
                    
                    # Find the vertical extent of lane line in this column
                    ll_ys = np.where(col_ll > 0)[0]
                    da_ys = np.where(col_da > 0)[0]
                    
                    if len(ll_ys) > 0 and len(da_ys) > 0:
                        # Check vertical span of lane line
                        ll_span = np.max(ll_ys) - np.min(ll_ys)
                        da_span = len(da_ys)
                        
                        # Only extend if lane line already spans a significant vertical distance
                        # (at least 30% of the drivable area height in this column)
                        if ll_span > 0 and da_span > 0:
                            span_ratio = ll_span / da_span
                            if span_ratio > 0.3:
                                # This is a strong vertical lane line - extend it fully
                                da_min = np.min(da_ys)
                                da_max = np.max(da_ys)
                                ll_extended[da_min:da_max+1, x] = 255
                
                # Apply morphological operations to create solid dividers
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
                ll_dividers = cv2.morphologyEx(ll_extended, cv2.MORPH_CLOSE, kernel_close)
                
                # Dilate to make boundaries
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                ll_dividers = cv2.dilate(ll_dividers, kernel_dilate, iterations=1)
                
                # Step 2: Create segmentation by subtracting lane line dividers from drivable area
                segmentation_base = da_mask.copy().astype(np.uint8)
                segmentation_base[ll_dividers > 0] = 0  # Lane lines become boundaries
                
                # Step 3: Label connected regions (each lane segment gets a unique ID)
                num_labels, labels = cv2.connectedComponents(segmentation_base)
                
                # Step 4: Find the centermost lane (closest to image center horizontally)
                image_center_x = w_im // 2
                lane_centers = {}
                
                for label_id in range(1, num_labels):
                    mask = labels == label_id
                    if np.any(mask):
                        # Find horizontal center of this lane segment
                        ys, xs = np.where(mask)
                        lane_center_x = np.mean(xs)
                        lane_centers[label_id] = lane_center_x
                
                # Find centermost lane
                center_lane_id = None
                min_dist = float('inf')
                for label_id, lane_x in lane_centers.items():
                    dist = abs(lane_x - image_center_x)
                    if dist < min_dist:
                        min_dist = dist
                        center_lane_id = label_id
                
                # Step 5: Color each segment based on distance from center lane
                overlay = np.zeros_like(im_infer)
                
                # Define colors by distance (0=center, 1=adjacent, 2=further, 3=furthest)
                distance_colors = [
                    (0, 255, 0),      # 0: Green - center lane
                    (0, 255, 255),    # 1: Yellow - adjacent lanes
                    (0, 165, 255),    # 2: Orange - further lanes
                    (0, 0, 255),      # 3+: Red - furthest lanes
                ]
                
                # Calculate center lane position
                center_lane_x = lane_centers.get(center_lane_id, image_center_x) if center_lane_id else image_center_x
                
                # Color each segment based on horizontal distance from center
                for label_id, lane_x in lane_centers.items():
                    mask = labels == label_id
                    
                    if label_id == center_lane_id:
                        # Center lane is always green
                        color = distance_colors[0]
                    else:
                        # Calculate approximate lane distance
                        # Assume average lane width of ~80-100 pixels
                        avg_lane_width = 90
                        distance_from_center = abs(lane_x - center_lane_x) / avg_lane_width
                        distance_idx = min(int(round(distance_from_center)), len(distance_colors) - 1)
                        color = distance_colors[distance_idx]
                    
                    overlay[mask] = color
                
                # Store which lane the car is actually in for boundary detection
                car_bottom_y = h_im - 20
                car_label = labels[car_bottom_y, car_center_x] if labels[car_bottom_y, car_center_x] > 0 else 0
                
                # Blend overlay with original image
                da_indices = da_mask > 0
                if np.any(da_indices):
                    alpha = 0.35
                    im_infer[da_indices] = cv2.addWeighted(
                        im_infer[da_indices], 1.0 - alpha, 
                        overlay[da_indices], alpha, 0
                    )
                
                # Step 6: Draw original lane lines in yellow (for visibility)
                ll_indices = ll_mask > 0
                if np.any(ll_indices):
                    im_infer[ll_indices] = (0, 255, 255)
                
                # Step 7: Find and draw the boundaries of the current lane in blue
                left_poly = None
                right_poly = None
                
                if car_label > 0:
                    current_lane_mask = (labels == car_label).astype(np.uint8) * 255
                    
                    # Apply morphological closing to fill vertical gaps and ensure continuity
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
                    current_lane_mask = cv2.morphologyEx(current_lane_mask, cv2.MORPH_CLOSE, kernel_close)
                    
                    # Find left and right edges of current lane
                    left_edge_x = []
                    left_edge_y = []
                    right_edge_x = []
                    right_edge_y = []
                    
                    # Start from higher up in the image for more complete lane coverage
                    start_y = int(h_im * 0.3)
                    for y in range(start_y, h_im, 1):  # Sample every row for continuity
                        row = current_lane_mask[y, :]
                        if np.any(row > 0):
                            lane_xs = np.where(row > 0)[0]
                            if len(lane_xs) > 0:
                                left_boundary = np.min(lane_xs)
                                right_boundary = np.max(lane_xs)
                                
                                # Only add if reasonably on the correct side
                                if left_boundary < car_center_x:
                                    left_edge_x.append(left_boundary)
                                    left_edge_y.append(y)
                                if right_boundary > car_center_x:
                                    right_edge_x.append(right_boundary)
                                    right_edge_y.append(y)
                    
                    # Fit smooth polynomials for lane boundaries
                    if len(left_edge_x) >= 20:  # Lower threshold for better detection
                        left_edge_x = np.array(left_edge_x)
                        left_edge_y = np.array(left_edge_y)
                        weights = np.exp((left_edge_y - h_im) / 60.0)
                        try:
                            left_poly = np.polyfit(left_edge_y, left_edge_x, 2, w=weights)
                        except:
                            pass
                    
                    if len(right_edge_x) >= 20:  # Lower threshold for better detection
                        right_edge_x = np.array(right_edge_x)
                        right_edge_y = np.array(right_edge_y)
                        weights = np.exp((right_edge_y - h_im) / 60.0)
                        try:
                            right_poly = np.polyfit(right_edge_y, right_edge_x, 2, w=weights)
                        except:
                            pass
                    
                    # Temporal smoothing of polynomials (critical for stability)
                    if left_poly is not None:
                        left_poly_sm = smooth_path(left_poly, state['left_poly_history'], max_history=12)
                    else:
                        # If we lose the lane, keep using the smoothed history for persistence
                        if len(state['left_poly_history']) > 0:
                            left_poly_sm = np.mean(state['left_poly_history'], axis=0)
                        else:
                            left_poly_sm = None
                        
                    if right_poly is not None:
                        right_poly_sm = smooth_path(right_poly, state['right_poly_history'], max_history=12)
                    else:
                        if len(state['right_poly_history']) > 0:
                            right_poly_sm = np.mean(state['right_poly_history'], axis=0)
                        else:
                            right_poly_sm = None

                    # Draw smoothed lane boundaries (thick blue lines) over full lane height
                    # Determine actual extent of current lane for plotting range
                    lane_ys = np.where(current_lane_mask > 0)[0]
                    if len(lane_ys) > 0:
                        plot_y_min = max(int(np.min(lane_ys)), 0)
                        plot_y_max = min(int(np.max(lane_ys)), h_im - 1)
                    else:
                        plot_y_min = int(h_im * 0.3)
                        plot_y_max = h_im - 1
                    
                    plot_y = np.linspace(plot_y_min, plot_y_max, num=200).astype(np.int32)
                    
                    if left_poly_sm is not None:
                        left_plot_x = np.polyval(left_poly_sm, plot_y)
                        left_plot_x = np.clip(left_plot_x, 0, INFER_WIDTH - 1)
                        left_pts = np.array([np.vstack([left_plot_x, plot_y]).T], dtype=np.int32)
                        cv2.polylines(im_infer, left_pts, isClosed=False, color=(255, 0, 0), thickness=6)
                    
                    if right_poly_sm is not None:
                        right_plot_x = np.polyval(right_poly_sm, plot_y)
                        right_plot_x = np.clip(right_plot_x, 0, INFER_WIDTH - 1)
                        right_pts = np.array([np.vstack([right_plot_x, plot_y]).T], dtype=np.int32)
                        cv2.polylines(im_infer, right_pts, isClosed=False, color=(255, 0, 0), thickness=6)

                    # Lane Departure Warning: check if car is drifting from lane center
                    ldw_triggered = False
                    if left_poly_sm is not None and right_poly_sm is not None:
                        # Evaluate lane boundaries at bottom of frame
                        eval_y = h_im - 50
                        left_x_bottom = np.polyval(left_poly_sm, eval_y)
                        right_x_bottom = np.polyval(right_poly_sm, eval_y)
                        lane_center = (left_x_bottom + right_x_bottom) / 2
                        
                        # Check drift from current car center
                        drift = abs(car_center_x - lane_center)
                        if drift > LDW_MAX_DRIFT:
                            ldw_triggered = True

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
