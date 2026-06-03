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
LDW_CALIBRATION_SECONDS = 10
LDW_MIN_CALIBRATION_SAMPLES = 60
HOOD_CALIBRATION_FRAMES = 12

# --- State ---
latest_web_frame = None
latest_lane_overlay = None
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
    "car_center_x": INFER_WIDTH // 2,
    "calibrate_center_requested": False,
    "calibration_center_frames_left": 0,
    "calib_center_history": [],
    "ldw_calibration": None,
    "hood_y": None,
    "hood_row_history": [],
    "hood_detection_frames_left": HOOD_CALIBRATION_FRAMES
}

if os.path.exists('models/calibration.json'):
    try:
        with open('models/calibration.json', 'r') as f:
            calib_data = json.load(f)
            if "vp_x" in calib_data and "vp_y" in calib_data:
                state["calibration"] = {"vp_x": calib_data["vp_x"], "vp_y": calib_data["vp_y"]}
                print("[INFO] Successfully loaded Stable VP Calibration.", flush=True)
            if "ldw_calibration" in calib_data:
                state["ldw_calibration"] = calib_data["ldw_calibration"]
                print("[INFO] Loaded LDW baseline calibration.", flush=True)
            if "hood_y" in calib_data:
                state["hood_y"] = int(calib_data["hood_y"])
                state["hood_detection_frames_left"] = 0
                print(f"[INFO] Loaded hood line: y={state['hood_y']}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load calibration matrix: {e}", flush=True)

video_writer = None

def save_calibration_state():
    calib_data = {}
    if state.get("calibration"):
        calib_data.update(state["calibration"])
    if state.get("ldw_calibration"):
        calib_data["ldw_calibration"] = state["ldw_calibration"]
    if state.get("hood_y") is not None:
        calib_data["hood_y"] = int(state["hood_y"])

    os.makedirs('models', exist_ok=True)
    with open('models/calibration.json', 'w') as f:
        json.dump(calib_data, f)

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

def remove_small_lane_components(ll_mask, min_area=20, min_height=12):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((ll_mask > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(ll_mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        if area >= min_area and height >= min_height:
            cleaned[labels == label_id] = 255
    return cleaned

def get_drivable_row_bounds(da_mask):
    row_bounds = []
    for y in range(da_mask.shape[0]):
        xs = np.where(da_mask[y] > 0)[0]
        if len(xs) > 0:
            row_bounds.append((int(xs[0]), int(xs[-1])))
        else:
            row_bounds.append(None)
    return row_bounds

def estimate_hood_line(da_mask):
    if da_mask is None or not np.any(da_mask > 0):
        return None

    h_im, w_im = da_mask.shape[:2]
    x_min = int(w_im * 0.2)
    x_max = int(w_im * 0.8)
    bottom_edges = []

    for x in range(x_min, x_max):
        ys = np.where(da_mask[:, x] > 0)[0]
        if len(ys) > 0:
            bottom_edges.append(int(np.max(ys)))

    if len(bottom_edges) < max(20, int((x_max - x_min) * 0.1)):
        return None

    return int(round(np.percentile(bottom_edges, 75)))

def update_hood_line(da_mask):
    if da_mask is None or state["hood_detection_frames_left"] <= 0:
        return

    hood_y = estimate_hood_line(da_mask)
    if hood_y is None:
        return

    state["hood_row_history"].append(hood_y)
    state["hood_detection_frames_left"] -= 1

    if state["hood_detection_frames_left"] <= 0:
        state["hood_y"] = int(round(np.median(state["hood_row_history"])))
        state["hood_row_history"] = []
        save_calibration_state()

def apply_hood_cutoff(da_mask, hood_y):
    if da_mask is None or hood_y is None:
        return da_mask
    clipped_mask = da_mask.copy()
    cutoff_y = int(np.clip(hood_y, 0, da_mask.shape[0] - 1))
    if cutoff_y + 1 < clipped_mask.shape[0]:
        clipped_mask[cutoff_y + 1:, :] = 0
    return clipped_mask

def find_seed_labels(labels, row_bounds, center_x, seed_row, band_half_height=10):
    h_im, w_im = labels.shape
    left_hits = []
    right_hits = []

    for y in range(max(0, seed_row - band_half_height), min(h_im, seed_row + band_half_height + 1)):
        bounds = row_bounds[y]
        if bounds is None:
            continue
        left_bound, right_bound = bounds
        center_clamped = int(np.clip(center_x, left_bound, right_bound))

        for x in range(center_clamped, left_bound - 1, -1):
            label_id = int(labels[y, x])
            if label_id > 0:
                left_hits.append(label_id)
                break

        for x in range(center_clamped, right_bound + 1):
            label_id = int(labels[y, x])
            if label_id > 0:
                right_hits.append(label_id)
                break

    def choose_label(hit_list, banned_label=None):
        if not hit_list:
            return 0
        label_counts = {}
        for label_id in hit_list:
            if banned_label is not None and label_id == banned_label:
                continue
            label_counts[label_id] = label_counts.get(label_id, 0) + 1
        if not label_counts:
            return 0
        return max(label_counts.items(), key=lambda item: item[1])[0]

    left_label = choose_label(left_hits)
    right_label = choose_label(right_hits, banned_label=left_label)
    return left_label, right_label

def build_current_lane_line_masks(ll_mask, da_mask, center_x):
    if ll_mask is None or da_mask is None:
        return None, None, 0

    row_bounds = get_drivable_row_bounds(da_mask)
    drivable_rows = [y for y, bounds in enumerate(row_bounds) if bounds is not None]
    if not drivable_rows:
        return np.zeros_like(ll_mask), np.zeros_like(ll_mask), 0

    seed_row = drivable_rows[min(len(drivable_rows) - 1, int(len(drivable_rows) * 0.42))]
    ll_clean = cv2.morphologyEx(ll_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
    ll_clean = remove_small_lane_components(ll_clean, min_area=20, min_height=10)

    _, labels = cv2.connectedComponents((ll_clean > 0).astype(np.uint8), connectivity=8)
    left_label, right_label = find_seed_labels(labels, row_bounds, center_x, seed_row)

    left_mask = np.zeros_like(ll_mask)
    right_mask = np.zeros_like(ll_mask)
    if left_label > 0:
        left_mask[labels == left_label] = 255
    if right_label > 0:
        right_mask[labels == right_label] = 255

    return left_mask, right_mask, seed_row

def sample_line_position(line_mask, target_y, side):
    if line_mask is None or not np.any(line_mask > 0):
        return None

    target_y = int(np.clip(target_y, 0, line_mask.shape[0] - 1))
    xs = np.where(line_mask[target_y] > 0)[0]
    if len(xs) == 0:
        rows = np.where(np.any(line_mask > 0, axis=1))[0]
        if len(rows) == 0:
            return None
        nearest_row = int(rows[np.argmin(np.abs(rows - target_y))])
        xs = np.where(line_mask[nearest_row] > 0)[0]
        if len(xs) == 0:
            return None

    if side == 'left':
        return float(np.max(xs))
    return float(np.min(xs))

def finalize_ldw_calibration():
    left_samples = np.array(state["calib_left_history"], dtype=np.float32)
    right_samples = np.array(state["calib_right_history"], dtype=np.float32)
    eval_samples = np.array(state["calib_center_history"], dtype=np.float32)

    state["calibration_center_frames_left"] = 0
    state["calibrate_center_requested"] = False

    if len(left_samples) < LDW_MIN_CALIBRATION_SAMPLES or len(right_samples) < LDW_MIN_CALIBRATION_SAMPLES:
        state["ldw_calibration"] = None
        state["calib_left_history"] = []
        state["calib_right_history"] = []
        state["calib_center_history"] = []
        state["error"] = "Center calibration failed: not enough lane samples during the 10-second window."
        return

    left_min = float(np.percentile(left_samples, 10) - 8)
    left_max = float(np.percentile(left_samples, 90) + 8)
    right_min = float(np.percentile(right_samples, 10) - 8)
    right_max = float(np.percentile(right_samples, 90) + 8)
    eval_y = int(round(float(np.median(eval_samples)))) if len(eval_samples) > 0 else int(INFER_HEIGHT * 0.6)

    state["ldw_calibration"] = {
        "left_min": left_min,
        "left_max": left_max,
        "right_min": right_min,
        "right_max": right_max,
        "eval_y": eval_y,
        "sample_count": int(min(len(left_samples), len(right_samples)))
    }
    state["calib_left_history"] = []
    state["calib_right_history"] = []
    state["calib_center_history"] = []
    state["error"] = None
    save_calibration_state()

def smooth_track(track, history, max_history=8):
    if track is None:
        return None
    if history and np.shape(history[0]) != np.shape(track):
        history.clear()
    history.append(track)
    if len(history) > max_history:
        history.pop(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(history, axis=0)

def _estimate_track_slope(sample_ys, sample_xs, use_top, window_size=10):
    if len(sample_ys) < 2:
        return 0.0
    if use_top:
        ys = sample_ys[:window_size]
        xs = sample_xs[:window_size]
    else:
        ys = sample_ys[-window_size:]
        xs = sample_xs[-window_size:]
    if len(ys) < 2:
        return 0.0
    coeffs = np.polyfit(ys, xs, 1)
    return float(coeffs[0])

def build_lane_track(mask, row_bounds, side, smoothing_window=11):
    if mask is None or not np.any(mask > 0):
        return []

    track = np.full(mask.shape[0], np.nan, dtype=np.float32)
    observed_ys = []
    observed_xs = []
    drivable_ys = [y for y, bounds in enumerate(row_bounds) if bounds is not None]
    if not drivable_ys:
        return None

    for y in drivable_ys:
        row_xs = np.where(mask[y] > 0)[0]
        if len(row_xs) == 0:
            continue
        x_val = float(np.max(row_xs)) if side == 'left' else float(np.min(row_xs))
        observed_ys.append(y)
        observed_xs.append(x_val)

    if len(observed_xs) < 6:
        return None

    observed_ys = np.array(observed_ys, dtype=np.int32)
    observed_xs = np.array(observed_xs, dtype=np.float32)
    first_y = int(observed_ys[0])
    last_y = int(observed_ys[-1])

    interp_ys = np.arange(first_y, last_y + 1, dtype=np.int32)
    interp_xs = np.interp(interp_ys, observed_ys, observed_xs)

    if len(interp_xs) >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float32) / float(smoothing_window)
        padded = np.pad(interp_xs, (smoothing_window // 2, smoothing_window // 2), mode='edge')
        interp_xs = np.convolve(padded, kernel, mode='valid')

    track[interp_ys] = interp_xs.astype(np.float32)

    top_slope = _estimate_track_slope(interp_ys, interp_xs, use_top=True)
    bottom_slope = _estimate_track_slope(interp_ys, interp_xs, use_top=False)

    top_drivable_y = int(drivable_ys[0])
    bottom_drivable_y = int(drivable_ys[-1])
    for y in range(first_y - 1, top_drivable_y - 1, -1):
        track[y] = float(track[y + 1] - top_slope)
    for y in range(last_y + 1, bottom_drivable_y + 1):
        track[y] = float(track[y - 1] + bottom_slope)

    for y in drivable_ys:
        if np.isnan(track[y]):
            continue
        left_bound, right_bound = row_bounds[y]
        track[y] = float(np.clip(track[y], left_bound, right_bound))

    return track

def curve_to_points(track, row_bounds):
    if track is None:
        return []

    points = []
    for y, bounds in enumerate(row_bounds):
        if bounds is None or np.isnan(track[y]):
            continue
        points.append([round(float(track[y]), 2), int(y)])
    return points

def build_lane_overlay_payload(left_mask, right_mask, da_mask, row_bounds, left_warning=False, right_warning=False, fcw_warning=False):
    h_im, w_im = da_mask.shape[:2]
    left_track = build_lane_track(left_mask, row_bounds, 'left')
    right_track = build_lane_track(right_mask, row_bounds, 'right')

    left_track_sm = None
    right_track_sm = None
    if left_track is not None:
        left_track_sm = smooth_track(left_track, state['left_poly_history'], max_history=8)
    elif len(state['left_poly_history']) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            left_track_sm = np.nanmean(state['left_poly_history'], axis=0)

    if right_track is not None:
        right_track_sm = smooth_track(right_track, state['right_poly_history'], max_history=8)
    elif len(state['right_poly_history']) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            right_track_sm = np.nanmean(state['right_poly_history'], axis=0)

    left_points = curve_to_points(left_track_sm, row_bounds) if left_track_sm is not None else []
    right_points = curve_to_points(right_track_sm, row_bounds) if right_track_sm is not None else []
    center_points = []
    if left_track_sm is not None and right_track_sm is not None:
        for y, bounds in enumerate(row_bounds):
            if bounds is None or np.isnan(left_track_sm[y]) or np.isnan(right_track_sm[y]):
                continue
            center_x = 0.5 * (left_track_sm[y] + right_track_sm[y])
            center_points.append([round(float(center_x), 2), int(y)])

    polygon = []
    if len(left_points) >= 2 and len(right_points) >= 2:
        polygon = left_points + list(reversed(right_points))

    left_zone = []
    right_zone = []
    if len(left_points) >= 2 and len(center_points) >= 2:
        left_zone = left_points + list(reversed(center_points))
    if len(center_points) >= 2 and len(right_points) >= 2:
        right_zone = center_points + list(reversed(right_points))

    drivable_rows = [y for y, bounds in enumerate(row_bounds) if bounds is not None]
    top_y = drivable_rows[0] if drivable_rows else 0
    bottom_y = drivable_rows[-1] if drivable_rows else h_im - 1

    return {
        "width": w_im,
        "height": h_im,
        "top_y": int(top_y),
        "bottom_y": int(bottom_y),
        "left_points": left_points,
        "right_points": right_points,
        "center_points": center_points,
        "polygon": polygon,
        "left_zone": left_zone,
        "right_zone": right_zone,
        "left_warning": bool(left_warning),
        "right_warning": bool(right_warning),
        "fcw_warning": bool(fcw_warning),
    }

def inference_loop():
    global latest_web_frame, latest_lane_overlay, raw_frame_buffer, state
    
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
        lane_overlay_payload = None

        lane_perception_active = state["adas_enabled"] or state["calibration_center_frames_left"] > 0

        if state["adas_enabled"]:
            # --- 1. YOLOv8 Forward Collision Warning ---
            # Predict objects without tracking overhead
            results = yolo_model.predict(im_infer, classes=[2, 5, 7], verbose=False, device=yolo_dev) # Cars, buses, trucks
            hood_y = state.get("hood_y")
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    cx = (x1 + x2) / 2

                    # Ignore detections that fall into the hood zone below the learned road boundary.
                    if hood_y is not None and y2 >= hood_y:
                        continue
                    
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

        if lane_perception_active:
            # --- 2. TwinLiteNet Drivable Area & Lane Departure Warning ---
            da_mask, ll_mask = twinlite_model.detect(im_infer)

            if ll_mask is not None and da_mask is not None:
                update_hood_line(da_mask)
                hood_y = state.get("hood_y")
                da_lane_mask = apply_hood_cutoff(da_mask, hood_y)
                car_center_x = state.get("car_center_x", INFER_WIDTH // 2)
                h_im, w_im = im_infer.shape[:2]
                row_bounds = get_drivable_row_bounds(da_lane_mask)
                left_line_mask, right_line_mask, seed_row = build_current_lane_line_masks(ll_mask, da_lane_mask, car_center_x)
                left_out_of_range = False
                right_out_of_range = False

                cv2.line(im_infer, (car_center_x, 0), (car_center_x, h_im - 1), (255, 255, 255), 1)
                if hood_y is not None:
                    cv2.line(im_infer, (0, int(hood_y)), (w_im - 1, int(hood_y)), (255, 0, 255), 2)

                # Keep the drivable area lightly visible, but only segment the current left/right lane lines.
                da_indices = da_lane_mask > 0
                if np.any(da_indices):
                    overlay = np.zeros_like(im_infer)
                    overlay[da_indices] = (0, 120, 0)
                    alpha = 0.18
                    im_infer[da_indices] = cv2.addWeighted(
                        im_infer[da_indices], 1.0 - alpha,
                        overlay[da_indices], alpha, 0
                    )

                ll_indices = ll_mask > 0
                if np.any(ll_indices):
                    im_infer[ll_indices] = (0, 255, 255)

                cv2.line(im_infer, (0, seed_row), (w_im - 1, seed_row), (255, 255, 255), 1)

                eval_y = min(h_im - 1, seed_row + 80)
                left_x = sample_line_position(left_line_mask, eval_y, 'left')
                right_x = sample_line_position(right_line_mask, eval_y, 'right')

                if state["calibration_center_frames_left"] > 0:
                    state["calibration_center_frames_left"] -= 1
                    if left_x is not None and right_x is not None and right_x > left_x:
                        state["calib_left_history"].append(left_x)
                        state["calib_right_history"].append(right_x)
                        state["calib_center_history"].append(eval_y)

                    if state["calibration_center_frames_left"] <= 0:
                        finalize_ldw_calibration()

                ldw_calibration = state.get("ldw_calibration")
                if state["adas_enabled"] and ldw_calibration and left_x is not None and right_x is not None and right_x > left_x:
                    left_out_of_range = left_x < ldw_calibration["left_min"] or left_x > ldw_calibration["left_max"]
                    right_out_of_range = right_x < ldw_calibration["right_min"] or right_x > ldw_calibration["right_max"]
                    if left_out_of_range or right_out_of_range:
                        ldw_triggered = True

                    cv2.line(im_infer, (0, int(ldw_calibration["eval_y"])), (w_im - 1, int(ldw_calibration["eval_y"])), (0, 255, 255), 1)

                lane_overlay_payload = build_lane_overlay_payload(
                    left_line_mask,
                    right_line_mask,
                    da_lane_mask,
                    row_bounds,
                    left_warning=left_out_of_range,
                    right_warning=right_out_of_range,
                    fcw_warning=fcw_triggered,
                )

        # Update global state for UI alerts
        state["fcw_warning"] = fcw_triggered
        state["ldw_warning"] = ldw_triggered

        # Encode to JPEG for the web interface
        _, buf = cv2.imencode('.jpg', im_infer, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_web_frame = buf.tobytes()
            latest_lane_overlay = lane_overlay_payload

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

@app.route('/api/lane_overlay')
def lane_overlay():
    with frame_lock:
        overlay = latest_lane_overlay

    if overlay is None:
        return jsonify({
            "width": INFER_WIDTH,
            "height": INFER_HEIGHT,
            "top_y": 0,
            "bottom_y": INFER_HEIGHT - 1,
            "left_points": [],
            "right_points": [],
            "polygon": []
        })
    return jsonify(overlay)

@app.route('/api/status')
def status():
    duration = 0
    if state["recording"] and state["recording_since"]:
        duration = int(time.time() - state["recording_since"])
    calibration_total_frames = TARGET_FPS * LDW_CALIBRATION_SECONDS
    
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
        "calibration_center_progress": calibration_total_frames - state["calibration_center_frames_left"],
        "calibration_center_total": calibration_total_frames,
        "ldw_calibrated": state["ldw_calibration"] is not None
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
    state["calibrate_center_requested"] = False
    state["calibration_center_frames_left"] = TARGET_FPS * LDW_CALIBRATION_SECONDS
    state["calib_left_history"] = []
    state["calib_right_history"] = []
    state["calib_center_history"] = []
    state["ldw_calibration"] = None
    state["error"] = None
    return jsonify({"status": "calibrating"})

@app.route('/api/set_center_x', methods=['POST'])
def api_set_center_x():
    global state
    data = request.json or {}
    center_x = data.get('car_center_x')
    if center_x is None:
        return jsonify({"success": False, "error": "car_center_x is required"}), 400

    state["car_center_x"] = int(np.clip(int(center_x), 0, INFER_WIDTH - 1))
    save_calibration_state()
    return jsonify({"success": True, "car_center_x": state["car_center_x"]})

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
