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

# Import YolopDetector for unified GPU inference
from yolop_detector import YolopDetector

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
LDW_MIN_CALIBRATION_SAMPLES = 30
LDW_WARNING_CONSECUTIVE_FRAMES = 3
HOOD_CALIBRATION_FRAMES = 12
LENS_CALIBRATION_FRAMES = 30

# --- Road Plane / Bird's Eye View (BEV) Geometry Configuration ---
CAMERA_HEIGHT = 1.2192 # 4 feet off the ground in meters
BEV_WIDTH = 240
BEV_HEIGHT = 1980
X_MIN = -6.0
X_MAX = 6.0
Y_MIN = 1.0
Y_MAX = 100.0

# --- Camera Calibration Matrix and Distortion Coefficients ---
def load_camera_calibration():
    yaml_path = 'calibration.yaml'
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(__file__), 'calibration.yaml')
        
    camera_matrix = None
    dist_coeff = None
    
    if os.path.exists(yaml_path):
        try:
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                camera_matrix = fs.getNode("camera_matrix").mat()
                dist_coeff = fs.getNode("dist_coeff").mat()
                fs.release()
                if camera_matrix is not None and dist_coeff is not None:
                    print(f"[INFO] Loaded camera calibration from {yaml_path}", flush=True)
                    return camera_matrix.astype(np.float32), dist_coeff.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to load calibration.yaml: {e}", flush=True)
            
    print("[WARNING] Using fallback camera calibration parameters", flush=True)
    camera_matrix = np.array([
        [898.6913680933326, 0.0, 673.4475138526925],
        [0.0, 898.7300068900809, 349.2407561225512],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_coeff = np.array([0.027602996212313838, -0.064486646048556584,
                           0.0034829585578156821, -0.0048244561182151577,
                           0.035676429431834245], dtype=np.float32)
    return camera_matrix, dist_coeff

CAMERA_MATRIX, DIST_COEFF = load_camera_calibration()

# Camera matrix scaled for inference size (640x400)
K_INFER = np.array([
    [CAMERA_MATRIX[0, 0] * 0.5, 0.0, CAMERA_MATRIX[0, 2] * 0.5],
    [0.0, CAMERA_MATRIX[1, 1] * 0.5, CAMERA_MATRIX[1, 2] * 0.5],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

H = None
H_inv = None
H_cam2bev = None
ROAD_TO_CAMERA_MATRIX = None


def undistort_image_points(pts):
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K_INFER, DIST_COEFF, P=K_INFER).reshape(-1, 2)

# --- State ---
latest_web_frame = None
latest_debug_frame = None
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
    "yolop_device": "Unknown",
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
    "calib_vp_x_list": [],
    "calib_vp_y_list": [],
    "calib_left_history": [],
    "calib_right_history": [],
    "calibration": None,
    "car_center_x": INFER_WIDTH // 2,
    "calibration_ldw_left_start_time": None,
    "calibration_ldw_right_start_time": None,
    "calib_ldw_left_dists": [],
    "calib_ldw_right_dists": [],
    "ldw_calibration": None,
    "ldw_left_counter": 0,
    "ldw_right_counter": 0,
    "hood_y": None,
    "hood_row_history": [],
    "hood_detection_frames_left": HOOD_CALIBRATION_FRAMES,
    "lane_position": 0.5,
    "lane_width": 0.0,
}

def update_homography(vp_override=None):
    global H, H_inv, H_cam2bev, ROAD_TO_CAMERA_MATRIX
    vp = vp_override if vp_override is not None else state.get("calibration")
    if vp is not None:
        vp_x = vp["vp_x"]
        vp_y = vp["vp_y"]
    else:
        vp_x = K_INFER[0, 2] # Default center
        vp_y = 160.0 # Default vanishing point y (approx 1.86 deg pitch down)

    vp_x, vp_y = undistort_image_points(np.array([[vp_x, vp_y]], dtype=np.float32))[0]
        
    K_inv = np.linalg.inv(K_INFER)
    
    # Forward unit vector in camera coordinates
    u_y = K_inv @ np.array([vp_x, vp_y, 1.0], dtype=np.float32)
    u_y = u_y / np.linalg.norm(u_y)
    
    # Right unit vector (horizontal in camera, roll=0)
    u_x = np.array([u_y[2], 0.0, -u_y[0]], dtype=np.float32)
    u_x = u_x / np.linalg.norm(u_x)
    
    # Plane normal points upward in the OpenCV camera frame.
    u_z = np.cross(u_x, u_y)
    if u_z[1] > 0:
        u_z = -u_z
        
    h = CAMERA_HEIGHT
    
    # The camera sits h meters above the road plane, so the plane-origin translation
    # in camera coordinates is opposite the plane normal direction.
    M = np.stack([u_x, u_y, -h * u_z], axis=1)
    ROAD_TO_CAMERA_MATRIX = M
    
    # Homography H mapping road plane (X, Y) to image coordinates
    H = K_INFER @ M
    H_inv = np.linalg.inv(H)
    
    # Mapping from road (X, Y) to BEV pixels
    s_x = (BEV_WIDTH - 1) / (X_MAX - X_MIN)
    t_x = -X_MIN * s_x
    s_y = (BEV_HEIGHT - 1) / (Y_MAX - Y_MIN)
    t_y = Y_MAX * s_y
    
    M_road2bev = np.array([
        [s_x, 0.0, t_x],
        [0.0, -s_y, t_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    H_cam2bev = M_road2bev @ H_inv

# Coordinate transformation helpers
def image_to_road(pts):
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts_undist = undistort_image_points(pts)
    pts_h = np.hstack([pts_undist, np.ones((len(pts_undist), 1), dtype=np.float32)])
    road_h = (H_inv @ pts_h.T).T
    valid = road_h[:, 2] > 1e-5
    road = np.zeros((len(pts_undist), 2), dtype=np.float32)
    road[valid, 0] = road_h[valid, 0] / road_h[valid, 2]
    road[valid, 1] = road_h[valid, 1] / road_h[valid, 2]
    road[~valid] = np.nan
    return road

def road_to_bev(pts):
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    s_x = (BEV_WIDTH - 1) / (X_MAX - X_MIN)
    s_y = (BEV_HEIGHT - 1) / (Y_MAX - Y_MIN)
    u_bev = (pts[:, 0] - X_MIN) * s_x
    v_bev = (Y_MAX - pts[:, 1]) * s_y
    return np.stack([u_bev, v_bev], axis=1)

def road_to_image(pts_road):
    if len(pts_road) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts_h = np.hstack([pts_road[:, 0:1], pts_road[:, 1:2], np.ones((len(pts_road), 1), dtype=np.float32)])
    cam_pts = (ROAD_TO_CAMERA_MATRIX @ pts_h.T).T
    valid = cam_pts[:, 2] > 1e-5
    img = np.zeros((len(pts_road), 2), dtype=np.float32)
    if np.any(valid):
        img_valid, _ = cv2.projectPoints(
            cam_pts[valid].reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            K_INFER,
            DIST_COEFF,
        )
        img[valid] = img_valid.reshape(-1, 2)
    img[~valid] = np.nan
    return img


def estimate_lane_lines_in_image(ll_mask, car_center_x):
    if ll_mask is None:
        return None

    row_step = 4
    max_jump_px = 36.0
    min_points = 8

    def collect_side_points(direction):
        points = []
        prev_u = None

        for v in range(INFER_HEIGHT - 1, max(120, int(INFER_HEIGHT * 0.35)), -row_step):
            row = ll_mask[v]
            cols = np.flatnonzero(row > 0)
            if len(cols) == 0:
                continue

            if direction == "left":
                cols = cols[cols < car_center_x]
                if len(cols) == 0:
                    continue
                candidate = cols[-1]
            else:
                cols = cols[cols > car_center_x]
                if len(cols) == 0:
                    continue
                candidate = cols[0]

            seg_left = candidate
            seg_right = candidate
            while seg_left - 1 >= 0 and row[seg_left - 1] > 0:
                seg_left -= 1
            while seg_right + 1 < len(row) and row[seg_right + 1] > 0:
                seg_right += 1

            segment_center = 0.5 * (seg_left + seg_right)
            if prev_u is not None and abs(segment_center - prev_u) > max_jump_px:
                continue

            points.append((float(segment_center), float(v)))
            prev_u = segment_center

        return np.array(points, dtype=np.float32) if len(points) >= min_points else None

    def fit_side_line(points):
        if points is None or len(points) < min_points:
            return None
        vs = points[:, 1]
        us = points[:, 0]
        try:
            line = np.polyfit(vs, us, 1)
            fit_us = np.polyval(line, vs)
            residuals = np.abs(us - fit_us)
            inliers = residuals < 8.0
            if np.sum(inliers) < min_points:
                return None
            line = np.polyfit(vs[inliers], us[inliers], 1)
            return (float(line[0]), float(line[1])), points[inliers]
        except Exception:
            return None

    left_points = collect_side_points("left")
    right_points = collect_side_points("right")
    left_fit = fit_side_line(left_points)
    right_fit = fit_side_line(right_points)
    if left_fit is None or right_fit is None:
        return None

    (m_left, c_left), left_points = left_fit
    (m_right, c_right), right_points = right_fit

    if abs(m_left - m_right) <= 0.05:
        return None

    v_intersect = (c_right - c_left) / (m_left - m_right)
    u_intersect = m_left * v_intersect + c_left
    if not (150 < u_intersect < 490 and 80 < v_intersect < 280):
        return None

    return {
        "vp_x": float(u_intersect),
        "vp_y": float(v_intersect),
        "left_line": (float(m_left), float(c_left)),
        "right_line": (float(m_right), float(c_right)),
        "left_points": left_points,
        "right_points": right_points,
    }


def estimate_vanishing_point_from_mask(ll_mask, car_center_x):
    lane_lines = estimate_lane_lines_in_image(ll_mask, car_center_x)
    if lane_lines is None:
        return None
    return float(lane_lines["vp_x"]), float(lane_lines["vp_y"])


def build_lane_rectification_homography(lane_lines):
    if lane_lines is None:
        return None

    m_left, c_left = lane_lines["left_line"]
    m_right, c_right = lane_lines["right_line"]
    vp_y = lane_lines["vp_y"]

    bottom_v = float(INFER_HEIGHT - 1)
    top_v = float(np.clip(vp_y + 24.0, 140.0, INFER_HEIGHT - 100.0))

    left_bottom = m_left * bottom_v + c_left
    right_bottom = m_right * bottom_v + c_right
    left_top = m_left * top_v + c_left
    right_top = m_right * top_v + c_right

    bottom_width = right_bottom - left_bottom
    top_width = right_top - left_top
    if bottom_width < 80.0 or top_width < 20.0:
        return None

    src = np.array([
        [left_bottom, bottom_v],
        [right_bottom, bottom_v],
        [right_top, top_v],
        [left_top, top_v],
    ], dtype=np.float32)

    dst_left = BEV_WIDTH * 0.33
    dst_right = BEV_WIDTH * 0.67
    dst = np.array([
        [dst_left, BEV_HEIGHT - 1],
        [dst_right, BEV_HEIGHT - 1],
        [dst_right, 0],
        [dst_left, 0],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src, dst)

if os.path.exists('models/calibration.json'):
    try:
        with open('models/calibration.json', 'r') as f:
            calib_data = json.load(f)
            if "vp_x" in calib_data and "vp_y" in calib_data:
                state["calibration"] = {"vp_x": calib_data["vp_x"], "vp_y": calib_data["vp_y"]}
                print("[INFO] Successfully loaded Stable VP Calibration.", flush=True)
            if "ldw_calibration" in calib_data:
                state["ldw_calibration"] = calib_data["ldw_calibration"]
                state["adas_enabled"] = True
                print("[INFO] Loaded LDW baseline calibration. Auto-enabling ADAS.", flush=True)
            if "hood_y" in calib_data:
                state["hood_y"] = int(calib_data["hood_y"])
                state["hood_detection_frames_left"] = 0
                print(f"[INFO] Loaded hood line: y={state['hood_y']}", flush=True)
            if "car_center_x" in calib_data:
                state["car_center_x"] = int(calib_data["car_center_x"])
                print(f"[INFO] Loaded car center bias: x={state['car_center_x']}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load calibration matrix: {e}", flush=True)

# Initialize homography
update_homography()

video_writer = None

def save_calibration_state():
    calib_data = {}
    if state.get("calibration"):
        calib_data.update(state["calibration"])
    if state.get("ldw_calibration"):
        calib_data["ldw_calibration"] = state["ldw_calibration"]
    if state.get("hood_y") is not None:
        calib_data["hood_y"] = int(state["hood_y"])
    if state.get("car_center_x") is not None:
        calib_data["car_center_x"] = int(state["car_center_x"])

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
        try:
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
        except Exception:
            logging.exception("capture_loop frame processing failed")
            time.sleep(0.05)


# --- Thread 2: AI Inference and Web Encoding ---
import warnings



def auto_calculate_lane_center_bias():
    ldw_calib = state.get("ldw_calibration")
    if not ldw_calib or not isinstance(ldw_calib, dict):
        return
    
    left_comfort = ldw_calib.get("left_comfort_dist")
    right_comfort = ldw_calib.get("right_comfort_dist")
    
    if left_comfort is not None and left_comfort > 10.0:
        left_comfort = 1.1
    if right_comfort is not None and right_comfort > 10.0:
        right_comfort = 1.1
    
    if left_comfort is not None and right_comfort is not None:
        # Calculate shift
        shift = (right_comfort - left_comfort) / 2.0
        
        # Update car_center_x
        old_center = state.get("car_center_x", INFER_WIDTH // 2)
        
        # Project old center to meters
        car_center_pts = np.array([[old_center, INFER_HEIGHT - 1]], dtype=np.float32)
        car_center_road = image_to_road(car_center_pts)[0]
        old_center_meters = car_center_road[0] if not np.isnan(car_center_road[0]) else 0.0
        
        new_center_meters = old_center_meters + shift
        
        # Project back to image pixels
        new_center_img = road_to_image(np.array([[new_center_meters, 1.0]]))[0]
        new_center = int(np.clip(new_center_img[0], 0, INFER_WIDTH - 1)) if not np.isnan(new_center_img[0]) else old_center
        
        # Calculate new balanced comfort distance
        balanced_comfort = (left_comfort + right_comfort) / 2.0
        
        state["car_center_x"] = new_center
        ldw_calib["left_comfort_dist"] = balanced_comfort
        ldw_calib["right_comfort_dist"] = balanced_comfort
        
        print(f"[INFO] Auto-balanced center bias: shifted from {old_center} to {new_center} (shift={shift:.2f} m). New balanced comfort dist: {balanced_comfort:.2f} m", flush=True)

def finalize_ldw_left_calibration():
    dists = np.array(state["calib_ldw_left_dists"], dtype=np.float32)
    state["calibration_ldw_left_start_time"] = None
    
    if len(dists) < LDW_MIN_CALIBRATION_SAMPLES:
        state["error"] = "Left LDW calibration failed: not enough lane samples."
        return
        
    left_comfort_dist = float(np.mean(dists))
    
    ldw_calib = state.get("ldw_calibration") or {}
    if not isinstance(ldw_calib, dict):
        ldw_calib = {}
        
    ldw_calib["left_comfort_dist"] = left_comfort_dist
    ldw_calib["eval_y"] = 5.0
    
    state["ldw_calibration"] = ldw_calib
    state["error"] = None
    auto_calculate_lane_center_bias()
    save_calibration_state()
    print(f"[INFO] Calibrated left comfortable distance: {left_comfort_dist:.2f} m", flush=True)

def finalize_ldw_right_calibration():
    dists = np.array(state["calib_ldw_right_dists"], dtype=np.float32)
    state["calibration_ldw_right_start_time"] = None
    
    if len(dists) < LDW_MIN_CALIBRATION_SAMPLES:
        state["error"] = "Right LDW calibration failed: not enough lane samples."
        return
        
    right_comfort_dist = float(np.mean(dists))
    
    ldw_calib = state.get("ldw_calibration") or {}
    if not isinstance(ldw_calib, dict):
        ldw_calib = {}
        
    ldw_calib["right_comfort_dist"] = right_comfort_dist
    ldw_calib["eval_y"] = 5.0
    
    state["ldw_calibration"] = ldw_calib
    state["error"] = None
    auto_calculate_lane_center_bias()
    save_calibration_state()
    print(f"[INFO] Calibrated right comfortable distance: {right_comfort_dist:.2f} m", flush=True)

def find_lanes_sliding_window(ll_mask_undist, bev_display_h, car_center_x_meters):
    if ll_mask_undist is None:
        return None, None

    ll_bev = cv2.warpPerspective(ll_mask_undist, bev_display_h, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_NEAREST)

    histogram = np.sum(ll_bev[int(BEV_HEIGHT*3/4):, :], axis=0)

    peaks = []
    threshold = np.max(histogram) * 0.1
    if threshold > 0:
        for i in range(1, BEV_WIDTH - 1):
            if histogram[i] > threshold and histogram[i] >= histogram[i-1] and histogram[i] >= histogram[i+1]:
                if not peaks or i - peaks[-1] > 30:
                    peaks.append(i)
                elif histogram[i] > histogram[peaks[-1]]:
                    peaks[-1] = i

    nwindows = 20
    window_height = int(BEV_HEIGHT / nwindows)
    margin = 40
    minpix = 8

    nonzero = ll_bev.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_points = {p: ([], []) for p in peaks}

    for peak in peaks:
        current_x = peak
        for window in range(nwindows):
            win_y_low = BEV_HEIGHT - (window + 1) * window_height
            win_y_high = BEV_HEIGHT - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            if len(good_inds) > 0:
                lane_points[peak][0].extend(nonzerox[good_inds])
                lane_points[peak][1].extend(nonzeroy[good_inds])

            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))

    s_x = (BEV_WIDTH - 1) / (X_MAX - X_MIN)
    s_y = (BEV_HEIGHT - 1) / (Y_MAX - Y_MIN)

    lane_polys = []
    for peak, (xs, ys) in lane_points.items():
        if len(xs) > 20:
            xs = np.array(xs)
            ys = np.array(ys)

            # Filter to inner edge of lane marking to avoid road edge bias
            median_x = np.median(xs)
            if peak > BEV_WIDTH // 2:
                # Right side: keep leftmost (innermost) half of points
                inner = xs < median_x
            else:
                # Left side: keep rightmost (innermost) half of points
                inner = xs > median_x
            if np.sum(inner) >= 10:
                xs = xs[inner]
                ys = ys[inner]
            else:
                continue

            road_Xs = xs / s_x + X_MIN
            road_Ys = Y_MAX - ys / s_y

            try:
                poly = np.polyfit(road_Ys, road_Xs, 2)
                fit_Xs = np.polyval(poly, road_Ys)
                residuals = np.abs(road_Xs - fit_Xs)
                inliers = residuals < 0.25
                if np.sum(inliers) >= 10:
                    poly = np.polyfit(road_Ys[inliers], road_Xs[inliers], 2)
                    lane_polys.append(poly)
            except Exception:
                pass

    left_poly = None
    right_poly = None
    eval_ys = [5.0, 15.0, 25.0]
    eval_weights = [0.5, 0.3, 0.2]

    best_left_score = -999.0
    best_right_score = 999.0

    for poly in lane_polys:
        weighted_diff = 0.0
        for ey, w in zip(eval_ys, eval_weights):
            x_at_eval = np.polyval(poly, ey)
            weighted_diff += w * (x_at_eval - car_center_x_meters)

        if weighted_diff < -0.3 and weighted_diff > best_left_score:
            best_left_score = weighted_diff
            left_poly = poly
        elif weighted_diff > 0.3 and weighted_diff < best_right_score:
            best_right_score = weighted_diff
            right_poly = poly

    if left_poly is not None and right_poly is not None:
        widths = []
        for ey in eval_ys:
            lx = np.polyval(left_poly, ey)
            rx = np.polyval(right_poly, ey)
            w = rx - lx
            if 2.0 < w < 6.0:
                widths.append(w)
        if len(widths) < 2:
            return None, None

    return left_poly, right_poly

def build_lane_overlay_payload(left_poly, right_poly, left_severity=0.0, right_severity=0.0, fcw_warning=False, fcw_boxes=None, lane_position=None, lane_width=None):
    if left_poly is None and right_poly is None:
        return {
            "width": BEV_WIDTH, "height": BEV_HEIGHT, "top_y": 0, "bottom_y": BEV_HEIGHT - 1,
            "left_points": [], "right_points": [], "center_points": [], "polygon": [],
            "left_zone": [], "right_zone": [],
            "left_severity": float(left_severity), "right_severity": float(right_severity),
            "fcw_warning": bool(fcw_warning), "fcw_boxes": fcw_boxes or [],
            "lane_position": lane_position if lane_position is not None else 0.5,
            "lane_width": lane_width if lane_width is not None else 0.0,
        }

    # Evaluate polynomials to build BEV points
    eval_ys = np.arange(Y_MIN, Y_MAX, 1.0, dtype=np.float32)
    
    if left_poly is not None:
        left_xs = np.polyval(left_poly, eval_ys)
        left_pts_road = np.stack([left_xs, eval_ys], axis=1)
        left_pts_bev = road_to_bev(left_pts_road)
        left_points = left_pts_bev.tolist()
    else:
        left_points = []
        left_xs = None
        
    if right_poly is not None:
        right_xs = np.polyval(right_poly, eval_ys)
        right_pts_road = np.stack([right_xs, eval_ys], axis=1)
        right_pts_bev = road_to_bev(right_pts_road)
        right_points = right_pts_bev.tolist()
    else:
        right_points = []
        right_xs = None
        
    center_points = []
    if left_xs is not None and right_xs is not None:
        center_xs = 0.5 * (left_xs + right_xs)
        center_pts_road = np.stack([center_xs, eval_ys], axis=1)
        center_pts_bev = road_to_bev(center_pts_road)
        center_points = center_pts_bev.tolist()

    polygon = []
    if len(left_points) >= 2 and len(right_points) >= 2:
        polygon = left_points + list(reversed(right_points))
        
    left_zone = []
    right_zone = []
    if len(left_points) >= 2 and len(center_points) >= 2:
        left_zone = left_points + list(reversed(center_points))
    if len(center_points) >= 2 and len(right_points) >= 2:
        right_zone = center_points + list(reversed(right_points))
        
    return {
        "width": BEV_WIDTH,
        "height": BEV_HEIGHT,
        "top_y": 0,
        "bottom_y": BEV_HEIGHT - 1,
        "left_points": left_points,
        "right_points": right_points,
        "center_points": center_points,
        "polygon": polygon,
        "left_zone": left_zone,
        "right_zone": right_zone,
        "left_severity": float(left_severity),
        "right_severity": float(right_severity),
        "fcw_warning": bool(fcw_warning),
        "fcw_boxes": fcw_boxes or [],
        "lane_position": lane_position if lane_position is not None else 0.5,
        "lane_width": lane_width if lane_width is not None else 0.0,
    }

def inference_loop():
    global latest_web_frame, latest_debug_frame, latest_lane_overlay, raw_frame_buffer, state
    
    fps_counter = 0
    fps_start = time.time()

    state["cuda_available"] = torch.cuda.is_available()
    state["gpu_device_name"] = torch.cuda.get_device_name(0) if state["cuda_available"] else "None"
    
    print("[INFO] Loading YOLOpv2 detector...", flush=True)
    yolop_detector = YolopDetector()
    state["yolop_device"] = str(yolop_detector.device)
    print(f"\n{'='*50}\n[DEBUG - GPU CHECK]\nCUDA Available: {state['cuda_available']}\nGPU Name: {state['gpu_device_name']}\nYOLOpv2 Device Target: {yolop_detector.device}\n{'='*50}\n", flush=True)

    while True:
        try:
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

            # Resize for inference (raw feed)
            im_infer = cv2.resize(im0, (INFER_WIDTH, INFER_HEIGHT), interpolation=cv2.INTER_LINEAR)
            im_debug = im_infer.copy()

            fcw_triggered = False
            ldw_triggered = False
            lane_overlay_payload = None
            fcw_overlay_boxes = []

            # Check if calibration requested
            if state.get("calibrate_requested"):
                state["calibration_frames_left"] = LENS_CALIBRATION_FRAMES
                state["calib_vp_x_list"] = []
                state["calib_vp_y_list"] = []
                state["error"] = None
                state["calibrate_requested"] = False

            car_center_x = state.get("car_center_x", INFER_WIDTH // 2)

            lane_perception_active = state["adas_enabled"] or state["calibration_frames_left"] > 0 or state.get("calibrate_requested", False)

            if state["adas_enabled"] or lane_perception_active:
                # Run YOLOpv2 inference
                det_boxes, da_mask, ll_mask = yolop_detector.detect(im_infer)
                
                # Undistort masks only for BEV warping. Point projection now undistorts on demand.
                da_mask_undist = (cv2.undistort((da_mask * 255).astype(np.uint8), K_INFER, DIST_COEFF) > 127).astype(np.uint8) if da_mask is not None else None
                ll_mask_undist = (cv2.undistort((ll_mask * 255).astype(np.uint8), K_INFER, DIST_COEFF) > 127).astype(np.uint8) if ll_mask is not None else None

                lane_lines_raw = estimate_lane_lines_in_image(ll_mask, car_center_x) if ll_mask is not None else None
                detected_vp = None if lane_lines_raw is None else (lane_lines_raw["vp_x"], lane_lines_raw["vp_y"])
                lane_lines_undist = estimate_lane_lines_in_image(ll_mask_undist, car_center_x) if ll_mask_undist is not None else None

                # Lens Calibration VP update
                if state["calibration_frames_left"] > 0 and detected_vp is not None:
                    state["calib_vp_x_list"].append(detected_vp[0])
                    state["calib_vp_y_list"].append(detected_vp[1])
                    state["calibration_frames_left"] -= 1
                    if state["calibration_frames_left"] == 0:
                        if len(state["calib_vp_x_list"]) >= 5:
                            vp_x = float(np.median(state["calib_vp_x_list"]))
                            vp_y = float(np.median(state["calib_vp_y_list"]))
                            state["calibration"] = {"vp_x": vp_x, "vp_y": vp_y}
                            state["error"] = None
                            print(f"[INFO] Calibrated vanishing point: ({vp_x:.2f}, {vp_y:.2f})", flush=True)
                            save_calibration_state()
                            update_homography()
                        else:
                            state["error"] = "Lens calibration failed: not enough lane lines detected."
                            print("[WARNING] Lens calibration failed", flush=True)

                car_center_pts = np.array([[car_center_x, INFER_HEIGHT - 1]], dtype=np.float32)
                car_center_road = image_to_road(car_center_pts)[0]
                car_center_x_meters = car_center_road[0] if not np.isnan(car_center_road[0]) else 0.0

                # Calculate BEV homography FIRST to pass to sliding window
                bev_display_h = build_lane_rectification_homography(lane_lines_undist)
                if bev_display_h is None:
                    bev_display_h = H_cam2bev

                # Fit lane lines in road plane
                left_poly, right_poly = None, None
                if ll_mask_undist is not None:
                    left_poly, right_poly = find_lanes_sliding_window(ll_mask_undist, bev_display_h, car_center_x_meters)

                # Smooth polynomials using history
                def smooth_poly(poly, history, max_history=8):
                    if poly is not None:
                        history.append(poly)
                        if len(history) > max_history:
                            history.pop(0)
                    if len(history) > 0:
                        return np.mean(history, axis=0)
                    return None

                left_poly_smoothed = smooth_poly(left_poly, state["left_poly_history"])
                right_poly_smoothed = smooth_poly(right_poly, state["right_poly_history"])

                # No fallback to straight lines if none detected

                # Project vehicles to road plane and calculate FCW
                if state["adas_enabled"] and det_boxes is not None:
                    for box in det_boxes:
                        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                        u_bot = (x1 + x2) / 2.0
                        v_bot = y2
                        
                        road_bot = image_to_road(np.array([[u_bot, v_bot]], dtype=np.float32))[0]
                        X_veh = road_bot[0]
                        Y_veh = road_bot[1]
                        
                        is_threat = False
                        if not np.isnan(X_veh) and not np.isnan(Y_veh):
                            # Ego lane check
                            if abs(X_veh - car_center_x_meters) < 1.6:
                                if Y_veh < 15.0:
                                    fcw_triggered = True
                                    is_threat = True
                            
                            # Construct BEV bounding box (width 1.8m, length 3.5m)
                            pts_veh_road = np.array([
                                [X_veh - 0.9, Y_veh + 3.5],
                                [X_veh + 0.9, Y_veh]
                            ], dtype=np.float32)
                            pts_veh_bev = road_to_bev(pts_veh_road)
                            x1_bev, y1_bev = pts_veh_bev[0]
                            x2_bev, y2_bev = pts_veh_bev[1]
                            
                            fcw_overlay_boxes.append({
                                "x1": round(float(x1_bev), 2),
                                "y1": round(float(y1_bev), 2),
                                "x2": round(float(x2_bev), 2),
                                "y2": round(float(y2_bev), 2),
                                "threat": is_threat,
                            })

                # Calculate LDW severities in road plane
                y_eval = 5.0
                
                if left_poly_smoothed is not None:
                    left_x_at_eval = np.polyval(left_poly_smoothed, y_eval)
                    d_left = car_center_x_meters - left_x_at_eval
                else:
                    d_left = 999.0
                    
                if right_poly_smoothed is not None:
                    right_x_at_eval = np.polyval(right_poly_smoothed, y_eval)
                    d_right = right_x_at_eval - car_center_x_meters
                else:
                    d_right = 999.0
                
                left_severity = 0.0
                right_severity = 0.0

                # 1. LDW Left Calibration
                if state.get("calibration_ldw_left_start_time") is not None:
                    elapsed = time.time() - state["calibration_ldw_left_start_time"]
                    if elapsed < LDW_CALIBRATION_SECONDS:
                        state["calib_ldw_left_dists"].append(float(d_left))
                    else:
                        finalize_ldw_left_calibration()

                # 2. LDW Right Calibration
                if state.get("calibration_ldw_right_start_time") is not None:
                    elapsed = time.time() - state["calibration_ldw_right_start_time"]
                    if elapsed < LDW_CALIBRATION_SECONDS:
                        state["calib_ldw_right_dists"].append(float(d_right))
                    else:
                        finalize_ldw_right_calibration()

                # Get comfortable thresholds in meters
                ldw_calibration = state.get("ldw_calibration")
                if not ldw_calibration:
                    left_comfort = 1.1
                    right_comfort = 1.1
                else:
                    left_comfort = ldw_calibration.get("left_comfort_dist", 1.1)
                    right_comfort = ldw_calibration.get("right_comfort_dist", 1.1)
                    if left_comfort > 10.0:
                        left_comfort = 1.1
                    if right_comfort > 10.0:
                        right_comfort = 1.1

                if state["adas_enabled"]:
                    if d_left <= left_comfort:
                        left_severity = (left_comfort - d_left) / 0.3
                        left_severity = float(np.clip(left_severity, 0.0, 1.0))
                    if d_right <= right_comfort:
                        right_severity = (right_comfort - d_right) / 0.3
                        right_severity = float(np.clip(right_severity, 0.0, 1.0))

                    if left_severity > 0.8 or right_severity > 0.8:
                        ldw_triggered = True

                # Render only the bottom 30m of the BEV map (600px)
                RENDER_H = 600
                im_bev = np.zeros((RENDER_H, BEV_WIDTH, 3), dtype=np.uint8)

                # Warp and draw drivable area mask in gray
                if da_mask_undist is not None:
                    da_bev_full = cv2.warpPerspective(da_mask_undist, bev_display_h, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_NEAREST)
                    da_bev = da_bev_full[-RENDER_H:]
                    im_bev[da_bev > 0] = (40, 40, 40) # Dark gray road

                # Draw clean vectorized ego lanes
                if left_poly_smoothed is not None and right_poly_smoothed is not None:
                    eval_ys = np.arange(1.0, 31.0, 1.0, dtype=np.float32)
                    left_xs = np.polyval(left_poly_smoothed, eval_ys)
                    right_xs = np.polyval(right_poly_smoothed, eval_ys)
                    
                    left_pts_road = np.stack([left_xs, eval_ys], axis=1)
                    right_pts_road = np.stack([right_xs, eval_ys], axis=1)
                    
                    left_pts_bev = road_to_bev(left_pts_road)
                    right_pts_bev = road_to_bev(right_pts_road)
                    
                    # Adjust for RENDER_H
                    left_pts_bev[:, 1] -= (BEV_HEIGHT - RENDER_H)
                    right_pts_bev[:, 1] -= (BEV_HEIGHT - RENDER_H)
                    
                    left_pts_int = left_pts_bev.astype(np.int32)
                    right_pts_int = right_pts_bev.astype(np.int32)
                    
                    # Fill ego lane with blue tint
                    ego_poly = np.vstack((left_pts_int, right_pts_int[::-1]))
                    overlay = im_bev.copy()
                    cv2.fillPoly(overlay, [ego_poly], (120, 60, 30)) # BGR for bluish tint
                    cv2.addWeighted(overlay, 0.4, im_bev, 0.6, 0, dst=im_bev)
                    
                    # Draw clean lane lines
                    cv2.polylines(im_bev, [left_pts_int], False, (220, 220, 220), 2, cv2.LINE_AA)
                    cv2.polylines(im_bev, [right_pts_int], False, (220, 220, 220), 2, cv2.LINE_AA)

                # Draw ego center guide and ego car
                car_center_bev_x = int(road_to_bev(np.array([[car_center_x_meters, 1.0]]))[0, 0])
                ego_corners_road = np.array([
                    [car_center_x_meters - 0.9, 1.0],
                    [car_center_x_meters - 0.9, 2.8],
                    [car_center_x_meters + 0.9, 2.8],
                    [car_center_x_meters + 0.9, 1.0]
                ], dtype=np.float32)
                ego_corners_bev = road_to_bev(ego_corners_road)
                ego_corners_bev[:, 1] -= (BEV_HEIGHT - RENDER_H)
                ego_corners_int = ego_corners_bev.astype(np.int32)

                cv2.line(im_bev, (car_center_bev_x, 0), (car_center_bev_x, RENDER_H - 1), (100, 100, 100), 1)
                
                cv2.fillPoly(im_bev, [ego_corners_int], (60, 60, 60))
                cv2.polylines(im_bev, [ego_corners_int], True, (255, 120, 0), 2)

                # Draw lane position indicator bar at top of BEV
                if left_poly_smoothed is not None and right_poly_smoothed is not None:
                    y_lane = 5.0
                    lx = np.polyval(left_poly_smoothed, y_lane)
                    rx = np.polyval(right_poly_smoothed, y_lane)
                    lw = rx - lx
                    if lw > 0.5:
                        pos = (car_center_x_meters - lx) / lw
                        pos = float(np.clip(pos, 0.0, 1.0))
                        bar_y = 8
                        bar_h = 8
                        bar_margin = 30
                        bar_left = bar_margin
                        bar_right = BEV_WIDTH - bar_margin
                        bar_mid = int(bar_left + pos * (bar_right - bar_left))
                        cv2.rectangle(im_bev, (bar_left, bar_y), (bar_right, bar_y + bar_h), (60, 60, 60), -1)
                        cv2.rectangle(im_bev, (bar_left, bar_y), (bar_right, bar_y + bar_h), (180, 180, 180), 1)
                        cv2.drawMarker(im_bev, (bar_mid, bar_y + bar_h // 2), (0, 255, 255),
                                       cv2.MARKER_TRIANGLE_DOWN, 10, 2)

                # Draw vehicles as red dots
                for box in fcw_overlay_boxes:
                    v_bev = box["y2"] # y2 corresponds to the bottom of the vehicle in full BEV
                    v_render = v_bev - (BEV_HEIGHT - RENDER_H)
                    if v_render >= 0 and v_render < RENDER_H:
                        u_bev = (box["x1"] + box["x2"]) / 2.0
                        cv2.circle(im_bev, (int(u_bev), int(v_render)), 6, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(im_bev, (int(u_bev), int(v_render)), 6, (255, 255, 255), 1, cv2.LINE_AA)

                # Calculate vehicle lane position (0=left line, 1=right line)
                lane_position = 0.5
                lane_width = 0.0
                if left_poly_smoothed is not None and right_poly_smoothed is not None:
                    y_lane = 5.0
                    lx = np.polyval(left_poly_smoothed, y_lane)
                    rx = np.polyval(right_poly_smoothed, y_lane)
                    lw = rx - lx
                    if lw > 0.5:
                        lane_width = lw
                        lane_position = (car_center_x_meters - lx) / lw
                        lane_position = float(np.clip(lane_position, 0.0, 1.0))
                state["lane_position"] = lane_position
                state["lane_width"] = lane_width

                # Build final payload
                lane_overlay_payload = build_lane_overlay_payload(
                    left_poly_smoothed,
                    right_poly_smoothed,
                    left_severity=left_severity,
                    right_severity=right_severity,
                    fcw_warning=fcw_triggered,
                    fcw_boxes=fcw_overlay_boxes,
                    lane_position=lane_position,
                    lane_width=lane_width,
                )
                
                # Draw debug overlays on im_debug
                im_debug = im_infer.copy()
                vp = state.get("calibration")
                if vp is not None:
                    cv2.circle(im_debug, (int(vp["vp_x"]), int(vp["vp_y"])), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(im_debug, (int(K_INFER[0, 2]), 160), 5, (0, 0, 255), -1)

                if da_mask is not None:
                    road_overlay = im_debug.copy()
                    road_overlay[da_mask > 0] = (0, 140, 0)
                    cv2.addWeighted(road_overlay, 0.28, im_debug, 0.72, 0, dst=im_debug)
                
                if ll_mask is not None:
                    im_debug[ll_mask > 0] = (0, 255, 255)

                if state["calibration_frames_left"] > 0 and lane_lines_raw is not None:
                    for line_key, color in (("left_line", (255, 0, 0)), ("right_line", (0, 0, 255))):
                        m_line, c_line = lane_lines_raw[line_key]
                        y0 = INFER_HEIGHT - 1
                        y1 = max(0, int(lane_lines_raw["vp_y"] + 24.0))
                        x0 = int(np.clip(m_line * y0 + c_line, 0, INFER_WIDTH - 1))
                        x1 = int(np.clip(m_line * y1 + c_line, 0, INFER_WIDTH - 1))
                        cv2.line(im_debug, (x0, y0), (x1, y1), color, 3)
                    
                if det_boxes is not None:
                    for box in det_boxes:
                        bx1, by1, bx2, by2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
                        cv2.rectangle(im_debug, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            else:
                # ADAS disabled fallback: just show blank BEV frame and ego-car
                im_bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
                car_center_bev_x = int(road_to_bev(np.array([[car_center_x_meters, 1.0]]))[0, 0])
                cv2.line(im_bev, (car_center_bev_x, 0), (car_center_bev_x, BEV_HEIGHT - 1), (100, 100, 100), 1)
                
                ego_corners_road = np.array([
                    [car_center_x_meters - 0.9, 1.0],
                    [car_center_x_meters - 0.9, 2.8],
                    [car_center_x_meters + 0.9, 2.8],
                    [car_center_x_meters + 0.9, 1.0]
                ], dtype=np.float32)
                ego_corners_bev = road_to_bev(ego_corners_road).astype(np.int32)
                
                overlay = im_bev.copy()
                cv2.fillPoly(overlay, [ego_corners_bev], (60, 60, 60))
                cv2.polylines(overlay, [ego_corners_bev], True, (255, 120, 0), 2)
                cv2.addWeighted(overlay, 0.6, im_bev, 0.4, 0, dst=im_bev)

            # Update global state for UI alerts
            state["fcw_warning"] = fcw_triggered
            state["ldw_warning"] = ldw_triggered

            # Encode to JPEG for the web interface
            _, buf = cv2.imencode('.jpg', im_bev, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _, buf_debug = cv2.imencode('.jpg', im_debug, [cv2.IMWRITE_JPEG_QUALITY, 60])
            with frame_lock:
                latest_web_frame = buf.tobytes()
                latest_debug_frame = buf_debug.tobytes()
                latest_lane_overlay = lane_overlay_payload

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                state["web_fps"] = round(fps_counter / elapsed, 1)
                fps_counter = 0
                fps_start = time.time()
        except Exception:
            logging.exception("inference_loop frame processing failed")
            time.sleep(0.05)


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

def generate_debug_mjpeg():
    while True:
        with frame_lock:
            frame = latest_debug_frame
        if frame is None:
            wait_img = np.zeros((400, 640, 3), dtype=np.uint8)
            cv2.putText(wait_img, "Initializing Debug Feed...",
                        (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            _, buf = cv2.imencode('.jpg', wait_img)
            frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.route('/debug_feed')
def debug_feed():
    return Response(generate_debug_mjpeg(),
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
    
    now = time.time()
    
    calibrating_ldw_left = False
    calibration_ldw_left_progress = 0
    if state.get("calibration_ldw_left_start_time") is not None:
        elapsed = now - state["calibration_ldw_left_start_time"]
        if elapsed < LDW_CALIBRATION_SECONDS:
            calibrating_ldw_left = True
            calibration_ldw_left_progress = int((elapsed / LDW_CALIBRATION_SECONDS) * 100)
        else:
            calibration_ldw_left_progress = 100

    calibrating_ldw_right = False
    calibration_ldw_right_progress = 0
    if state.get("calibration_ldw_right_start_time") is not None:
        elapsed = now - state["calibration_ldw_right_start_time"]
        if elapsed < LDW_CALIBRATION_SECONDS:
            calibrating_ldw_right = True
            calibration_ldw_right_progress = int((elapsed / LDW_CALIBRATION_SECONDS) * 100)
        else:
            calibration_ldw_right_progress = 100

    calibrating_lens = state["calibration_frames_left"] > 0 or state.get("calibrate_requested", False)
    calibration_lens_progress = 0
    if calibrating_lens:
        calibration_lens_progress = int(((LENS_CALIBRATION_FRAMES - state["calibration_frames_left"]) / LENS_CALIBRATION_FRAMES) * 100)
        calibration_lens_progress = int(np.clip(calibration_lens_progress, 0, 100))
            
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
        "yolop_device": state["yolop_device"],
        "car_center_x": state["car_center_x"],
        "calibrating_lens": calibrating_lens,
        "calibration_lens_progress": calibration_lens_progress,
        "calibrating_ldw_left": calibrating_ldw_left,
        "calibration_ldw_left_progress": calibration_ldw_left_progress,
        "calibrating_ldw_right": calibrating_ldw_right,
        "calibration_ldw_right_progress": calibration_ldw_right_progress,
        "calibration_total": 100,
        "lens_calibrated": state["calibration"] is not None,
        "ldw_calibrated": state["ldw_calibration"] is not None,
        "ldw_calibrated_left": state["ldw_calibration"] is not None and "left_comfort_dist" in state["ldw_calibration"],
        "ldw_calibrated_right": state["ldw_calibration"] is not None and "right_comfort_dist" in state["ldw_calibration"],
        "lane_position": state.get("lane_position", 0.5),
        "lane_width": state.get("lane_width", 0.0),
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
    state["calib_vp_x_list"] = []
    state["calib_vp_y_list"] = []
    state["calibration_frames_left"] = 0
    state["error"] = None
    return jsonify({"status": "calibrating"})

@app.route('/api/calibrate_ldw_left', methods=['POST'])
def api_calibrate_ldw_left():
    global state
    state["calibration_ldw_left_start_time"] = time.time()
    state["calib_ldw_left_dists"] = []
    state["calibration_ldw_right_start_time"] = None
    state["error"] = None
    return jsonify({"status": "calibrating_left"})

@app.route('/api/calibrate_ldw_right', methods=['POST'])
def api_calibrate_ldw_right():
    global state
    state["calibration_ldw_right_start_time"] = time.time()
    state["calib_ldw_right_dists"] = []
    state["calibration_ldw_left_start_time"] = None
    state["error"] = None
    return jsonify({"status": "calibrating_right"})

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
