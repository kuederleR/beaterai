import sys
import os
import json
import logging
import time
import threading
import datetime
import cv2
import numpy as np

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
logging.getLogger('werkzeug').setLevel(logging.ERROR)

os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)

VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', '/dev/video0')
DEV_VIDEO_PATH = os.environ.get('DEV_VIDEO_PATH', None)
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 800
TARGET_FPS = 30

BEV_DISPLAY_WIDTH = 500
BEV_DISPLAY_HEIGHT = 500
BEV_RENDER_METERS = 50.0

FCW_WARNING_DISTANCE = 15.0
LANE_HALF_WIDTH = 1.6

LDW_CALIBRATION_SECONDS = 10
LDW_MIN_CALIBRATION_SAMPLES = 30

from yolop_detector import YolopDetector

INFER_WIDTH = 640
DEBUG_SCALE = 0.5  # 1280×800 → 640×400 for overlay/encode
INFER_HEIGHT = 416
TRACE_START_Y = 370

BEV_GRID_SIZE = 100
BEV_X_RANGE = (-30.0, 30.0)
BEV_Y_RANGE = (-15.0, 15.0)


BEV_WIDTH = 240
BEV_HEIGHT = 1001
CAMERA_HEIGHT = 1.2192
X_MIN, X_MAX = -6.0, 6.0
Y_MIN, Y_MAX = 1.0, 51.0

CAMERA_MATRIX = None
DIST_COEFF = None

yaml_path = 'calibration.yaml'
if not os.path.exists(yaml_path):
    yaml_path = os.path.join(os.path.dirname(__file__), 'calibration.yaml')
if os.path.exists(yaml_path):
    try:
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            CAMERA_MATRIX = fs.getNode("camera_matrix").mat()
            DIST_COEFF = fs.getNode("dist_coeff").mat()
            fs.release()
    except Exception:
        pass

K_INFER = None
if CAMERA_MATRIX is not None:
    try:
        K_INFER = np.array([
            [CAMERA_MATRIX[0, 0] * 0.5, 0.0, CAMERA_MATRIX[0, 2] * 0.5],
            [0.0, CAMERA_MATRIX[1, 1] * 0.5, CAMERA_MATRIX[1, 2] * 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    except Exception:
        pass
if K_INFER is None:
    K_INFER = np.array([
        [898.6913680933326 * 0.5, 0.0, 673.4475138526925 * 0.5],
        [0.0, 898.7300068900809 * 0.5, 349.2407561225512 * 0.5],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

# ── Road-plane geometry (homography + lane projection) ────────────────────
H_inv = None
_road_to_bev_M = None
_fallback_vp = None
_bev_warp_M = None

def _fallback_compute_homography(vp_x=None, vp_y=None):
    global H_inv, _road_to_bev_M, _fallback_vp
    if vp_x is None and _fallback_vp is not None:
        vp_x, vp_y = _fallback_vp
    if vp_x is None:
        vp_x = K_INFER[0, 2]
        vp_y = 160.0

    K_inv = np.linalg.inv(K_INFER)
    u_y = K_inv @ np.array([vp_x, vp_y, 1.0], dtype=np.float32)
    u_y = u_y / np.linalg.norm(u_y)
    u_x = np.array([u_y[2], 0.0, -u_y[0]], dtype=np.float32)
    u_x = u_x / np.linalg.norm(u_x)
    u_z = np.cross(u_x, u_y)
    if u_z[1] > 0:
        u_z = -u_z
    M = np.stack([u_x, u_y, -CAMERA_HEIGHT * u_z], axis=1)
    H = K_INFER @ M
    H_inv = np.linalg.inv(H)
    s_x = (BEV_WIDTH - 1) / (X_MAX - X_MIN)
    t_x = -X_MIN * s_x
    s_y = (BEV_HEIGHT - 1) / (Y_MAX - Y_MIN)
    t_y = Y_MAX * s_y
    M_road2bev = np.array([
        [s_x, 0.0, t_x],
        [0.0, -s_y, t_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    _road_to_bev_M = M_road2bev @ H_inv

def _fallback_image_to_road(pts):
    if len(pts) == 0 or H_inv is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    road_h = (H_inv @ pts_h.T).T
    valid = road_h[:, 2] > 1e-5
    road = np.zeros((len(pts), 2), dtype=np.float32)
    road[valid, 0] = road_h[valid, 0] / road_h[valid, 2]
    road[valid, 1] = road_h[valid, 1] / road_h[valid, 2]
    road[~valid] = np.nan
    return road

def _fallback_road_to_bev_display(pts_road, disp_w, disp_h, disp_max_y=30.0):
    if _road_to_bev_M is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts_h = np.hstack([pts_road, np.ones((len(pts_road), 1), dtype=np.float32)])
    bev = (_road_to_bev_M @ pts_h.T).T
    valid = bev[:, 2] > 1e-5
    out = np.zeros((len(pts_road), 2), dtype=np.float32)
    u = bev[valid, 0] / bev[valid, 2]
    v = bev[valid, 1] / bev[valid, 2]
    s_x = disp_w / BEV_WIDTH
    s_y = disp_h / (BEV_HEIGHT * (disp_max_y / Y_MAX))
    out[valid, 0] = u * s_x
    out[valid, 1] = (v - (BEV_HEIGHT - disp_h)) * s_y if v.ndim == 0 else (v - (BEV_HEIGHT - disp_h))
    return out

def _trace_lane_edges(ll_mask, car_center_x, start_y, max_gap=30):
    h, w = ll_mask.shape[:2]
    start_y = int(np.clip(start_y, 10, h - 1))
    end_row = h // 3

    # Scan rows from bottom up, collecting left/right edge pixels
    left_pts = []
    right_pts = []
    last_left_x = None
    last_right_x = None
    left_gap = 0
    right_gap = 0

    for row in range(start_y, end_row - 1, -1):
        rd = ll_mask[row, :]
        nz = np.flatnonzero(rd)

        if len(nz) == 0:
            left_gap += 1
            right_gap += 1
            if left_gap > max_gap:
                left_pts.clear()
                last_left_x = None
            if right_gap > max_gap:
                right_pts.clear()
                last_right_x = None
            continue

        # Left edge: rightmost pixel left of center
        left_nz = nz[nz < car_center_x]
        if len(left_nz) > 0:
            left_gap = 0
            ex = int(left_nz[-1])
            left_pts.append((float(ex), float(row)))
            last_left_x = ex
        # Right edge: leftmost pixel right of center
        right_nz = nz[nz > car_center_x]
        if len(right_nz) > 0:
            right_gap = 0
            ex = int(right_nz[0])
            right_pts.append((float(ex), float(row)))
            last_right_x = ex

    def _to_array(pts):
        return np.array(pts, dtype=np.float32) if len(pts) >= 4 else None

    return _to_array(left_pts), _to_array(right_pts)


_lane_smooth_left = {}
_lane_smooth_right = {}
LANE_SMOOTH_ALPHA = 0.3

def _compute_lane_boundaries(da_mask, ll_mask, car_center_x, start_y):
    h, w = da_mask.shape[:2]
    start_y = int(np.clip(start_y, 10, h - 1))
    end_row = h // 3

    left_xs = []
    right_xs = []
    rows = []

    for row in range(start_y, end_row - 1, -1):
        da_row = da_mask[row, :]
        ll_row = ll_mask[row, :]

        da_nz = np.flatnonzero(da_row)
        if len(da_nz) < 2:
            continue
        da_left = da_nz[0]
        da_right = da_nz[-1]

        ll_nz = np.flatnonzero(ll_row)

        left_x = da_left
        if len(ll_nz) > 0:
            cand = ll_nz[(ll_nz < car_center_x) & (ll_nz >= da_left)]
            if len(cand) > 0:
                left_x = cand[-1]

        right_x = da_right
        if len(ll_nz) > 0:
            cand = ll_nz[(ll_nz > car_center_x) & (ll_nz <= da_right)]
            if len(cand) > 0:
                right_x = cand[0]

        if left_x < right_x:
            rows.append(row)
            left_xs.append(left_x)
            right_xs.append(right_x)

    if len(rows) < 2:
        return None, None

    alpha = LANE_SMOOTH_ALPHA
    smooth_l = []
    smooth_r = []
    for i, row in enumerate(rows):
        raw_l = float(left_xs[i])
        raw_r = float(right_xs[i])
        prev_l = _lane_smooth_left.get(row, raw_l)
        prev_r = _lane_smooth_right.get(row, raw_r)
        sm_l = alpha * raw_l + (1 - alpha) * prev_l
        sm_r = alpha * raw_r + (1 - alpha) * prev_r
        _lane_smooth_left[row] = sm_l
        _lane_smooth_right[row] = sm_r
        smooth_l.append(sm_l)
        smooth_r.append(sm_r)

    row_arr = np.array(rows, dtype=np.float32)
    left_edge = np.column_stack([np.array(smooth_l, dtype=np.float32), row_arr])
    right_edge = np.column_stack([np.array(smooth_r, dtype=np.float32), row_arr])

    return left_edge, right_edge


def _fallback_detect_lanes(ll_mask, car_center_x, trace_start_y):
    return _trace_lane_edges(ll_mask, car_center_x, trace_start_y)

def _fallback_fit_lanes(left_edge, right_edge, car_center_x):
    def _do_fit(pts):
        if pts is None or len(pts) < 4:
            return None
        road = _fallback_image_to_road(pts)
        valid = ~np.isnan(road[:, 0]) & ~np.isnan(road[:, 1])
        road = road[valid]
        if len(road) < 4:
            return None
        Y, X = road[:, 1], road[:, 0]
        try:
            poly = np.polyfit(Y, X, 2)
            return poly
        except np.linalg.LinAlgError:
            return None

    left = _do_fit(left_edge)
    right = _do_fit(right_edge)
    return left, right


def _bev_px_to_road(pts_bev, w, h):
    if len(pts_bev) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = (pts_bev[:, 0] / w - 0.5) * 2 * BEV_X_RANGE[1]
    y = (1.0 - pts_bev[:, 1] / h) * BEV_Y_RANGE[1]
    return np.column_stack([x, y])


def _bev_detect_lanes(ll_mask, cam_shape=None):
    global _bev_warp_M
    if _bev_warp_M is None:
        return [], None

    w = BEV_DISPLAY_WIDTH
    h = BEV_DISPLAY_HEIGHT

    ll_resized = ll_mask
    if cam_shape is not None:
        cw, ch = cam_shape[1], cam_shape[0]
        if ll_mask.shape[0] != ch or ll_mask.shape[1] != cw:
            ll_resized = cv2.resize(ll_mask, (cw, ch),
                                    interpolation=cv2.INTER_NEAREST)

    bev_mask = cv2.warpPerspective(
        (ll_resized * 255).astype(np.uint8), _bev_warp_M, (w, h),
        flags=cv2.INTER_NEAREST,
    )
    _, bev_mask = cv2.threshold(bev_mask, 127, 1, cv2.THRESH_BINARY)

    hist = np.sum(bev_mask[h // 2:, :], axis=0).astype(np.float32)
    kernel = np.ones(5, dtype=np.float32) / 5
    hist_smooth = np.convolve(hist, kernel, mode='same')

    thresh = max(np.max(hist_smooth) * 0.25, 3.0)
    min_dist = 15

    peaks = []
    for i in range(1, w - 1):
        if hist_smooth[i] > thresh and hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            if all(abs(i - p) >= min_dist for p in peaks):
                peaks.append(i)
    peaks.sort()

    win_w = 24
    win_h = 16
    y_step = 12
    y_lo = h // 4

    lanes = []
    all_attempts = []
    for bx in peaks:
        pts = []
        cx = float(bx)
        x1_base = max(0, int(cx) - win_w // 2)
        x2_base = min(w, int(cx) + win_w // 2)

        y_start = h - 1
        for yr in range(h - 1, y_lo, -1):
            if np.any(bev_mask[yr, x1_base:x2_base] > 0):
                y_start = yr
                break

        wins = []
        for y in range(y_start, y_lo, -y_step):
            y1 = max(0, y - win_h // 2)
            y2 = min(h, y + win_h // 2)
            x1 = max(0, int(cx) - win_w // 2)
            x2 = min(w, int(cx) + win_w // 2)
            strip = bev_mask[y1:y2, x1:x2]
            col_sums = np.sum(strip, axis=0).astype(np.float32)
            total = np.sum(col_sums)
            if total == 0:
                wins.append((x1, y1, x2, y2, None, None))
                break
            cols = np.arange(len(col_sums), dtype=np.float32)
            x_centroid = np.sum(cols * col_sums) / total
            mx = x1 + x_centroid
            if abs(mx - cx) > win_w * 0.6:
                wins.append((x1, y1, x2, y2, None, None))
                break
            row_sums = np.sum(strip, axis=1).astype(np.float32)
            rows = np.arange(len(row_sums), dtype=np.float32)
            y_centroid = np.sum(rows * row_sums) / np.sum(row_sums)
            my = y1 + y_centroid
            wins.append((x1, y1, x2, y2, mx, my))
            pts.append([mx, my])
            cx = mx

        # Trim bottom 3 points where warp distortion is worst
        if len(pts) > 6:
            pts = pts[3:]
        tracked = len(pts) >= 4
        if tracked:
            lanes.append(np.array(pts, dtype=np.float32))
        all_attempts.append({
            "peak": bx, "windows": wins,
            "lane_pts": pts, "tracked": tracked,
            "y_start": y_start,
        })

    return lanes, {
        "bev_mask": bev_mask,
        "peaks": peaks,
        "attempts": all_attempts,
    }


latest_web_frame = None
latest_debug_frame = None
_perspective_overlay_data = {
    "width": 640, "height": 400,
    "lanes": [],
    "vehicles": [],
}
raw_frame_buffer = None
calibration_frame = None
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
    "lane_position": 0.5,
    "lane_width": 0.0,
    "debug_feed_active": True,
    "calibrate_requested": False,
    "calibration_frames_left": 0,
    "calib_vp_x_list": [],
    "calib_vp_y_list": [],
    "calibration": None,
    "calibration_ldw_left_start_time": None,
    "calibration_ldw_right_start_time": None,
    "calib_ldw_left_positions": [],
    "calib_ldw_right_positions": [],
    "ldw_calibration": None,
    "car_center_x": 640,
    "car_center_offset": 0.0,
    "left_poly_history": [],
    "right_poly_history": [],
    "last_frame_time": 0.0,
    "capture_last_frame_time": 0.0,
}

video_writer = None
_camera = None

def _try_load_calibration():
    calib_path = "models/calibration.json"
    if not os.path.exists(calib_path):
        return
    try:
        with open(calib_path, 'r') as f:
            data = json.load(f)
        if "vp_x" in data and "vp_y" in data:
            state["calibration"] = {"vp_x": data["vp_x"], "vp_y": data["vp_y"]}
            print("[INFO] Loaded VP calibration", flush=True)
        if "ldw_calibration" in data:
            state["ldw_calibration"] = data["ldw_calibration"]
            state["adas_enabled"] = True
            print("[INFO] Loaded LDW calibration", flush=True)
        if "car_center_x" in data:
            state["car_center_x"] = int(data["car_center_x"])
        if "drivable_y_cutoff" in data:
            state["drivable_y_cutoff"] = int(data["drivable_y_cutoff"])
            print(f"[INFO] Loaded drivable y cutoff: {data['drivable_y_cutoff']}", flush=True)
    except Exception as e:
        print(f"[WARN] Calibration load failed: {e}", flush=True)

_try_load_calibration()

def save_calibration_state():
    calib = {}
    if state.get("calibration"):
        calib.update(state["calibration"])
    if state.get("ldw_calibration"):
        calib["ldw_calibration"] = state["ldw_calibration"]
    if state.get("car_center_x") is not None:
        calib["car_center_x"] = int(state["car_center_x"])
    if state.get("drivable_y_cutoff") is not None:
        calib["drivable_y_cutoff"] = int(state["drivable_y_cutoff"])
    os.makedirs("models", exist_ok=True)
    with open("models/calibration.json", "w") as f:
        json.dump(calib, f)

def make_error_frame(message):
    img = np.zeros((480, 800, 3), dtype=np.uint8)
    cv2.putText(img, message, (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


class _CamWrapper:
    def __init__(self):
        self._dev_path = DEV_VIDEO_PATH if (DEV_VIDEO_PATH and os.path.exists(DEV_VIDEO_PATH)) else None
        self._open()
    def _open(self):
        if self._dev_path:
            self._cap = cv2.VideoCapture(self._dev_path)
        else:
            self._cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    def start(self):
        return self._cap is not None and self._cap.isOpened()
    def read(self):
        return self._cap.read()
    def seek_to_start(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    def release(self):
        self._cap.release()

def capture_loop():
    global raw_frame_buffer, video_writer, calibration_frame, _camera

    cap = _CamWrapper()
    _camera = cap

    if not cap.start():
        state["error"] = "Failed to open camera"
        with frame_lock:
            latest_web_frame = make_error_frame(state["error"])
        return

    fps_counter = 0
    fps_start = time.time()

    is_dev_video = bool(DEV_VIDEO_PATH and os.path.exists(DEV_VIDEO_PATH))

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                if is_dev_video:
                    cap.seek_to_start()
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                    print("[INFO] Looping dev video", flush=True)
                else:
                    time.sleep(0.01)
                    continue

            if is_dev_video:
                time.sleep(1.0 / TARGET_FPS)

            with frame_lock:
                raw_frame_buffer = frame.copy()
                calibration_frame = frame.copy()

            if state["recording"]:
                if video_writer is None:
                    os.makedirs("recordings", exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"recordings/dashcam_{ts}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video_writer = cv2.VideoWriter(fname, fourcc, TARGET_FPS,
                                                   (frame.shape[1], frame.shape[0]))
                    state["recording_since"] = time.time()
                    print(f"[INFO] Recording: {fname}", flush=True)
                video_writer.write(frame)
            else:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    state["recording_since"] = None

            state["capture_last_frame_time"] = time.time()
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                state["capture_fps"] = round(fps_counter / elapsed, 1)
                fps_counter = 0
                fps_start = time.time()

        except Exception:
            logging.exception("capture_loop failed")
            time.sleep(0.05)


class StepTimer:
    def __init__(self, window_size=30, log_interval=30):
        self.times = {}
        self.window_size = window_size
        self.log_interval = log_interval
        self.frame_count = 0

    def track(self, name, elapsed):
        self.times.setdefault(name, []).append(elapsed)
        if len(self.times[name]) > self.window_size:
            self.times[name].pop(0)

    def avg(self, name):
        vals = self.times.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def maybe_log(self):
        self.frame_count += 1
        if self.frame_count % self.log_interval != 0:
            return
        ordered = ["total", "preprocess", "inference", "lane_post", "fcw", "bev_render", "encode"]
        parts = [f"{n}: {self.avg(n)*1000:.1f}ms" for n in ordered if n in self.times]
        total_avg = self.avg("total")
        print(f"[PERF] {' | '.join(parts)} | total: {total_avg*1000:.1f}ms ({1.0/total_avg:.1f}fps)", flush=True)


def render_bev_frame(lane_mask, detections, left_coeffs, right_coeffs,
                     compensated_x, lane_position, lane_width,
                     fcw_warning, left_sev, right_sev,
                     bg_img=None, bev_debug=None, raw_lanes=None):
    w = BEV_DISPLAY_WIDTH
    h = BEV_DISPLAY_HEIGHT

    if bg_img is not None:
        im_bev = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)
        draw_lanes = False
        if bev_debug is not None:
            mask = bev_debug["bev_mask"]
            green = np.zeros_like(im_bev, dtype=np.uint8)
            green[mask > 0] = (0, 180, 0)
            cv2.addWeighted(green, 0.2, im_bev, 0.8, 0, dst=im_bev)
            for bx in bev_debug["peaks"]:
                cv2.line(im_bev, (bx, 0), (bx, h - 1), (100, 100, 255), 1)
            for attempt in bev_debug["attempts"]:
                color = (0, 255, 255) if attempt["tracked"] else (0, 100, 255)
                for x1, y1, x2, y2, mx, my in attempt["windows"]:
                    cv2.rectangle(im_bev, (x1, y1), (x2, y2), color, 1)
                    if mx is not None:
                        cv2.circle(im_bev, (int(mx), int(my)), 2, (0, 255, 0), -1)
    else:
        im_bev = np.zeros((h, w, 3), dtype=np.uint8)
        draw_lanes = True

    s_x = w / (2 * BEV_X_RANGE[1])
    s_y = h / BEV_Y_RANGE[1]

    if draw_lanes:
        if lane_mask is not None:
            mask_disp = cv2.resize(lane_mask, (BEV_GRID_SIZE, BEV_GRID_SIZE),
                                   interpolation=cv2.INTER_NEAREST)
            mask_disp = cv2.resize(mask_disp.astype(np.uint8) * 255, (w, h),
                                   interpolation=cv2.INTER_NEAREST)
            im_bev[mask_disp > 0] = (40, 60, 40)

    if left_coeffs is not None and right_coeffs is not None:
        eval_y = np.arange(1.0, BEV_RENDER_METERS, 1.0, dtype=np.float32)
        left_x = np.polyval(left_coeffs, eval_y)
        right_x = np.polyval(right_coeffs, eval_y)

        def road_to_bev_px(rx, ry):
            u = int((rx / (2 * BEV_X_RANGE[1]) + 0.5) * w)
            v = int((1.0 - ry / BEV_Y_RANGE[1]) * h)
            return u, v

        left_pts = []
        right_pts = []
        for ry, rx in zip(eval_y, left_x):
            u, v = road_to_bev_px(rx, ry)
            if 0 <= u < w and 0 <= v < h:
                left_pts.append([u, v])
        for ry, rx in zip(eval_y, right_x):
            u, v = road_to_bev_px(rx, ry)
            if 0 <= u < w and 0 <= v < h:
                right_pts.append([u, v])

        if draw_lanes and len(left_pts) >= 2 and len(right_pts) >= 2:
            left_arr = np.array(left_pts, dtype=np.int32)
            right_arr = np.array(right_pts, dtype=np.int32)
            poly = np.vstack((left_arr, right_arr[::-1]))
            overlay = im_bev.copy()
            cv2.fillPoly(overlay, [poly], (120, 60, 30))
            cv2.addWeighted(overlay, 0.4, im_bev, 0.6, 0, dst=im_bev)

        if raw_lanes is not None:
            for pts in raw_lanes:
                if len(pts) >= 2:
                    cv2.polylines(im_bev, [pts.astype(np.int32)],
                                  False, (220, 220, 220), 2, cv2.LINE_AA)

        if len(left_pts) >= 2:
            cv2.polylines(im_bev, [np.array(left_pts, dtype=np.int32)],
                          False, (200, 100, 0), 2, cv2.LINE_AA)
        if len(right_pts) >= 2:
            cv2.polylines(im_bev, [np.array(right_pts, dtype=np.int32)],
                          False, (200, 100, 0), 2, cv2.LINE_AA)

    if draw_lanes:
        car_u = int(((compensated_x / (2 * BEV_X_RANGE[1])) + 0.5) * w)
        car_v = int(h * (1.0 - 2.0 / BEV_Y_RANGE[1]))
        cv2.line(im_bev, (car_u, 0), (car_u, h - 1), (100, 100, 100), 1)

        ego_car = np.array([
            [car_u - 12, car_v - 8],
            [car_u - 12, car_v + 8],
            [car_u + 12, car_v + 8],
            [car_u + 12, car_v - 8],
        ], dtype=np.int32)
        cv2.fillPoly(im_bev, [ego_car], (60, 60, 60))
        cv2.polylines(im_bev, [ego_car], True, (255, 120, 0), 2)

        for det in detections:
            cx = det["center_x"]
            cy = det["center_y"]
            du = int(((cx / (2 * BEV_X_RANGE[1])) + 0.5) * w)
            dv = int((1.0 - cy / BEV_Y_RANGE[1]) * h)
            if 0 <= dv < h:
                color = (0, 0, 255) if det.get("threat", False) else (0, 200, 200)
                cv2.circle(im_bev, (du, dv), 5, color, -1, cv2.LINE_AA)
                cv2.circle(im_bev, (du, dv), 5, (255, 255, 255), 1, cv2.LINE_AA)

    if lane_width > 0.5:
        bar_y = 8
        bar_h = 8
        bar_margin = 40
        bar_left = bar_margin
        bar_right = w - bar_margin
        bar_mid = int(bar_left + lane_position * (bar_right - bar_left))
        cv2.rectangle(im_bev, (bar_left, bar_y), (bar_right, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(im_bev, (bar_left, bar_y), (bar_right, bar_y + bar_h), (180, 180, 180), 1)
        cv2.drawMarker(im_bev, (bar_mid, bar_y + bar_h // 2), (0, 255, 255),
                       cv2.MARKER_TRIANGLE_DOWN, 10, 2)

    if left_sev > 0:
        warn_w = int(w * min(left_sev, 1.0))
        cv2.rectangle(im_bev, (0, 0), (warn_w, h), (0, 0, 255), 2)
    if right_sev > 0:
        warn_w = int(w * min(right_sev, 1.0))
        cv2.rectangle(im_bev, (w - warn_w, 0), (w - 1, h), (0, 0, 255), 2)

    return im_bev


def build_lane_overlay_payload(left_coeffs, right_coeffs,
                               left_severity=0.0, right_severity=0.0,
                               fcw_warning=False, fcw_boxes=None,
                               lane_position=None, lane_width=None):
    return {
        "width": BEV_DISPLAY_WIDTH,
        "height": BEV_DISPLAY_HEIGHT,
        "top_y": 0,
        "bottom_y": BEV_DISPLAY_HEIGHT - 1,
        "left_points": [],
        "right_points": [],
        "center_points": [],
        "polygon": [],
        "left_zone": [],
        "right_zone": [],
        "left_severity": float(left_severity),
        "right_severity": float(right_severity),
        "fcw_warning": bool(fcw_warning),
        "fcw_boxes": fcw_boxes or [],
        "lane_position": lane_position if lane_position is not None else 0.5,
        "lane_width": lane_width if lane_width is not None else 0.0,
    }


def _draw_filtered_lanes_perspective(im, left_coeffs, right_coeffs, bev_warp_M, lane_position=0.5, lead_car_color=None):
    if bev_warp_M is None:
        return
    if left_coeffs is None and right_coeffs is None:
        return

    h_im, w_im = im.shape[:2]
    sx, sy = w_im / CAPTURE_WIDTH, h_im / CAPTURE_HEIGHT
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    M_inv = S @ np.linalg.inv(bev_warp_M)

    bw, bh = BEV_DISPLAY_WIDTH, BEV_DISPLAY_HEIGHT
    xr, yr = BEV_X_RANGE[1], BEV_Y_RANGE[1]
    y_vals = np.arange(2.0, 18.0, 0.5, dtype=np.float32)

    left_pts = right_pts = None
    if left_coeffs is not None:
        rx = np.polyval(left_coeffs, y_vals)
        u = ((rx / (2 * xr) + 0.5) * bw).astype(np.float32)
        v = ((1.0 - y_vals / yr) * bh).astype(np.float32)
        pts_bev = np.column_stack([u, v]).reshape(1, -1, 2).astype(np.float32)
        pts_cam = cv2.perspectiveTransform(pts_bev, M_inv).reshape(-1, 2).astype(np.int32)
        pts_cam[:, 0] = np.clip(pts_cam[:, 0], 0, w_im - 1)
        pts_cam[:, 1] = np.clip(pts_cam[:, 1], 0, h_im - 1)
        left_pts = pts_cam
    if right_coeffs is not None:
        rx = np.polyval(right_coeffs, y_vals)
        u = ((rx / (2 * xr) + 0.5) * bw).astype(np.float32)
        v = ((1.0 - y_vals / yr) * bh).astype(np.float32)
        pts_bev = np.column_stack([u, v]).reshape(1, -1, 2).astype(np.float32)
        pts_cam = cv2.perspectiveTransform(pts_bev, M_inv).reshape(-1, 2).astype(np.int32)
        pts_cam[:, 0] = np.clip(pts_cam[:, 0], 0, w_im - 1)
        pts_cam[:, 1] = np.clip(pts_cam[:, 1], 0, h_im - 1)
        right_pts = pts_cam

    # Lane area fill
    if left_pts is not None and right_pts is not None:
        min_len = min(len(left_pts), len(right_pts))
        poly = np.vstack([left_pts[:min_len], right_pts[:min_len][::-1]])
        if lead_car_color:
            area_color = (int(lead_car_color[5:7], 16), int(lead_car_color[3:5], 16), int(lead_car_color[1:3], 16))
        else:
            area_color = (255, 100, 0)
        ov = im.copy()
        cv2.fillPoly(ov, [poly], area_color)
        cv2.addWeighted(ov, 0.25, im, 0.75, 0, im)

    # Gradient lane lines
    lp_clamped = max(0.0, min(1.0, lane_position))
    left_shift = 2 * abs(lp_clamped - 0.5) if lp_clamped < 0.5 else 0.0
    right_shift = 2 * abs(lp_clamped - 0.5) if lp_clamped > 0.5 else 0.0
    for pts, shift in [(left_pts, left_shift), (right_pts, right_shift)]:
        if pts is None or len(pts) < 2:
            continue
        b = int(255 * (1 - shift))
        g = int(100 * (1 - shift))
        r = int(255 * shift)
        for i in range(len(pts) - 1):
            cv2.line(im, tuple(pts[i]), tuple(pts[i + 1]), (b, g, r), 3, cv2.LINE_AA)


def _road_to_image(road_pts):
    if H_inv is None:
        return None
    H = np.linalg.inv(H_inv)
    pts_h = np.hstack([road_pts, np.ones((len(road_pts), 1), dtype=np.float32)])
    img_h = (H @ pts_h.T).T
    valid = img_h[:, 2] > 1e-5
    out = np.zeros((len(road_pts), 2), dtype=np.float32)
    out[valid, 0] = img_h[valid, 0] / img_h[valid, 2]
    out[valid, 1] = img_h[valid, 1] / img_h[valid, 2]
    out[~valid] = np.nan
    return out


def _robust_polyfit(road_x, road_y, deg=2):
    if len(road_x) < deg + 2 or len(road_y) < deg + 2:
        return None
    p1 = np.polyfit(road_y, road_x, 1)
    pred = np.polyval(p1, road_y)
    residuals = np.abs(road_x - pred)
    median_res = np.median(residuals)
    if median_res < 1e-6:
        median_res = 1e-6
    mad = np.median(np.abs(residuals - median_res))
    inliers = residuals < median_res + 2 * mad
    n_inliers = np.sum(inliers)
    if n_inliers < deg + 2:
        return None
    return np.polyfit(road_y[inliers], road_x[inliers], deg)


class _SimpleMotionDetector:
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = np.empty((0, 2), dtype=np.float32)
        self._dy_mean = 0.0
        self._mean_alpha = 0.08
        self._moving = False
        self._dy_value = 0.0

    def is_moving(self):
        return self._moving

    def get_dy(self):
        return self._dy_value

    def update(self, frame_gray):
        h, w = frame_gray.shape
        roi_y1, roi_y2 = int(h * 0.55), int(h * 0.85)
        roi_x1, roi_x2 = int(w * 0.25), int(w * 0.75)

        if self.prev_gray is None or len(self.prev_pts) < 10:
            self.prev_gray = frame_gray.copy()
            roi = frame_gray[roi_y1:roi_y2, roi_x1:roi_x2]
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=200, qualityLevel=0.01,
                                          minDistance=5, blockSize=7)
            if pts is not None:
                pts[:, :, 0] += roi_x1
                pts[:, :, 1] += roi_y1
                self.prev_pts = pts.reshape(-1, 2).astype(np.float32)
            return

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray,
            self.prev_pts.reshape(-1, 1, 2),
            None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if next_pts is not None and status is not None:
            good = status.ravel() == 1
            n_g = np.sum(good)
            if n_g > 5:
                gp = self.prev_pts[good]
                gn = next_pts[good].reshape(-1, 2)
                dy = np.abs(gn[:, 1] - gp[:, 1])
                mdy = float(np.median(dy))
                self._dy_value = mdy

                self._dy_mean = self._mean_alpha * mdy + (1.0 - self._mean_alpha) * self._dy_mean

                thresh = max(self._dy_mean * 0.3, 2.0)
                self._moving = mdy > thresh

                self.prev_pts = gn[:200].copy()
            else:
                self.prev_pts = np.empty((0, 2), dtype=np.float32)
        else:
            self.prev_pts = np.empty((0, 2), dtype=np.float32)

        self.prev_gray = frame_gray.copy()


class _VehicleTracker:
    def __init__(self, association_dist=4.0, smooth_alpha=0.35, max_missed=8):
        self.tracks = {}
        self.next_id = 0
        self.association_dist = association_dist
        self.smooth_alpha = smooth_alpha
        self.max_missed = max_missed

    def update(self, detections):
        new_tracks = {}
        used = set()

        for det in detections:
            cx, cy = det["center_x"], det["center_y"]
            best_id = None
            best_dist = self.association_dist
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                d = np.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                tid = best_id
                t = self.tracks[tid]
                old_cy = t["cy"]
                t["cx"] = t["cx"] * (1 - self.smooth_alpha) + cx * self.smooth_alpha
                t["cy"] = t["cy"] * (1 - self.smooth_alpha) + cy * self.smooth_alpha
                t["prev_cy"] = old_cy
                t["missed"] = 0
                t["width"] = det.get("width", 0.5)
                t["length"] = det.get("length", 0.5)
                t["conf"] = det.get("conf", 0.0)
                t["img_x1"] = det.get("img_x1", 0)
                t["img_y1"] = det.get("img_y1", 0)
                t["img_x2"] = det.get("img_x2", 0)
                t["img_y2"] = det.get("img_y2", 0)
                used.add(tid)
                new_tracks[tid] = t
            else:
                tid = self.next_id
                self.next_id += 1
                new_tracks[tid] = {
                    "id": tid,
                    "cx": cx,
                    "cy": cy,
                    "prev_cy": cy,
                    "missed": 0,
                    "width": det.get("width", 0.5),
                    "length": det.get("length", 0.5),
                    "conf": det.get("conf", 0.0),
                    "img_x1": det.get("img_x1", 0),
                    "img_y1": det.get("img_y1", 0),
                    "img_x2": det.get("img_x2", 0),
                    "img_y2": det.get("img_y2", 0),
                }

        for tid, t in self.tracks.items():
            if tid not in new_tracks:
                t["missed"] += 1
                if t["missed"] <= self.max_missed:
                    new_tracks[tid] = t

        self.tracks = new_tracks

        result = []
        for tid, t in self.tracks.items():
            if t["missed"] > 0:
                continue
            result.append({
                "center_x": t["cx"],
                "center_y": t["cy"],
                "width": t.get("width", 0.5),
                "length": t.get("length", 0.5),
                "conf": t.get("conf", 0.0),
                "track_id": tid,
                "prev_cy": t.get("prev_cy", t["cy"]),
                "img_x1": t.get("img_x1", 0),
                "img_y1": t.get("img_y1", 0),
                "img_x2": t.get("img_x2", 0),
                "img_y2": t.get("img_y2", 0),
            })
        return result


motion = _SimpleMotionDetector()
vehicle_tracker = _VehicleTracker()


def inference_loop():
    global latest_web_frame, latest_debug_frame, raw_frame_buffer

    _left_smooth = None
    _right_smooth = None
    _left_stale = 0
    _right_stale = 0
    _left_history = []
    _right_history = []
    _HISTORY_MAX = 15
    _SNAP_CURVE_THRESHOLD = 0.003

    _expected_left_bx = None
    _expected_right_bx = None
    _expected_width_px = None
    _expected_left_missed = 0
    _expected_right_missed = 0
    _intersection_hold = 0

    print("[INFO] Loading YOLOPv2 model...", flush=True)
    trt_engine = os.environ.get("YOLOP_TRT_ENGINE")
    detector = YolopDetector(model_path="data/weights/yolopv2.pt",
                             trt_engine_path=trt_engine)
    _fallback_compute_homography()
    print("[INFO] ADAS pipeline ready (YOLOPv2 + homography)", flush=True)
    timer = StepTimer()

    fps_counter = 0
    fps_start = time.time()

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

            _t_frame = time.perf_counter()
            im_debug = im0.copy()

            if len(im0.shape) == 2:
                im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
                im_debug = im0.copy()
            elif len(im0.shape) == 3 and im0.shape[2] == 4:
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
                im_debug = im0.copy()

            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            motion.update(gray)

            if DEBUG_SCALE != 1.0:
                new_w = int(im_debug.shape[1] * DEBUG_SCALE)
                new_h = int(im_debug.shape[0] * DEBUG_SCALE)
                im_debug = cv2.resize(im_debug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            lane_mask = None
            detections = []
            left_coeffs = None
            right_coeffs = None
            lane_position = 0.5
            lane_width = 0.0
            fcw_triggered = False
            ldw_triggered = False
            left_severity = 0.0
            right_severity = 0.0
            fcw_overlay_boxes = []
            compensated_x = 0.0
            bev_debug = None
            lanes_bev = []

            if state["adas_enabled"]:
                _t_inf = time.perf_counter()

                det_boxes, da_mask, ll_mask = detector.detect(im0)
                detections = []
                if det_boxes:
                    for b in det_boxes:
                        cx = (b["x1"] + b["x2"]) / 2.0
                        by = b["y2"]
                        road_pt = _fallback_image_to_road(np.array([[cx, by]], dtype=np.float32))
                        if len(road_pt) > 0 and not np.isnan(road_pt[0, 0]):
                            rw = abs(b["x2"] - b["x1"]) * 0.15
                            rh = abs(b["y2"] - b["y1"]) * 0.08
                            detections.append({
                                "center_x": float(road_pt[0, 0]),
                                "center_y": float(road_pt[0, 1]),
                                "width": max(rw, 0.5),
                                "length": max(rh, 0.5),
                                "conf": b.get("conf", 0.0),
                                "img_x1": float(b["x1"]),
                                "img_y1": float(b["y1"]),
                                "img_x2": float(b["x2"]),
                                "img_y2": float(b["y2"]),
                            })
                detections = vehicle_tracker.update(detections)

                # Auto-calibrate drivable Y cutoff from da_mask center column
                if da_mask is not None and state.get("drivable_y_cutoff") is None:
                    center_col = da_mask.shape[1] // 2
                    da_center = da_mask[:, center_col]
                    drivable_rows = np.where(da_center > 0)[0]
                    if len(drivable_rows) > 0:
                        bottom_row = int(drivable_rows[-1])
                        samples = state.setdefault("calib_drivable_y_samples", [])
                        samples.append(bottom_row)
                        if len(samples) >= 30:
                            cutoff = int(np.median(samples))
                            state["drivable_y_cutoff"] = cutoff
                            state["calib_drivable_y_samples"] = []
                            print(f"[CALIB] Drivable Y cutoff: {cutoff}", flush=True)
                            save_calibration_state()

                if ll_mask is not None:
                    lane_mask = ll_mask
                    car_center_x = state.get("car_center_x", INFER_WIDTH // 2)
                    tr_start = int(ll_mask.shape[0] * TRACE_START_Y / 480)
                    left_edge, right_edge = _compute_lane_boundaries(da_mask, ll_mask, car_center_x, tr_start)
                    if left_edge is not None:
                        left_lane_pts = left_edge
                    else:
                        left_lane_pts = None
                    if right_edge is not None:
                        right_lane_pts = right_edge
                    else:
                        right_lane_pts = None
                else:
                    left_lane_pts = None
                    right_lane_pts = None

                timer.track("inference", time.perf_counter() - _t_inf)

                _t_lane = time.perf_counter()
                car_center_x = state.get("car_center_x", INFER_WIDTH // 2)

                if motion.is_moving():
                    max_a = 0.003
                else:
                    max_a = None

                if _bev_warp_M is not None and ll_mask is not None:
                    lanes_bev, bev_debug = _bev_detect_lanes(ll_mask, cam_shape=im0.shape[:2])
                    w_bev = BEV_DISPLAY_WIDTH
                    h_bev = BEV_DISPLAY_HEIGHT
                    car_center_bev = w_bev // 2

                    cand_list = []
                    for pts in lanes_bev:
                        if len(pts) < 3:
                            continue
                        bot = pts[np.argmax(pts[:, 1])]
                        bx = float(bot[0])
                        cand_list.append({"pts": pts, "bx": bx})

                    def _pick_innermost(cands, target_x, side, car_center, skip=None):
                        candidates = []
                        for c in cands:
                            if c is skip:
                                continue
                            bx = c["bx"]
                            if side == 'left' and bx >= car_center:
                                continue
                            if side == 'right' and bx < car_center:
                                continue
                            candidates.append(c)
                        if not candidates:
                            return None
                        # Among all candidates on the correct side, pick the innermost (closest to
                        # car center). Use distance to expected position as a tiebreaker only,
                        # so a drifted expected position cannot lock onto the wrong lane line.
                        if side == 'left':
                            candidates.sort(key=lambda x: (-x["bx"], abs(x["bx"] - target_x)))
                        else:
                            candidates.sort(key=lambda x: (x["bx"], abs(x["bx"] - target_x)))
                        return candidates[0]

                    left_cand = None
                    right_cand = None

                    if _expected_left_bx is not None:
                        left_cand = _pick_innermost(cand_list, _expected_left_bx, 'left', car_center_bev)
                    if _expected_right_bx is not None:
                        right_cand = _pick_innermost(cand_list, _expected_right_bx, 'right', car_center_bev)

                    # Ensure left != right
                    if left_cand is not None and right_cand is not None and left_cand is right_cand:
                        dl = abs(left_cand["bx"] - (_expected_left_bx or 0))
                        dr = abs(right_cand["bx"] - (_expected_right_bx or 0))
                        if dl <= dr:
                            right_cand = _pick_innermost(cand_list, _expected_right_bx, 'right', car_center_bev, skip=left_cand)
                        else:
                            left_cand = _pick_innermost(cand_list, _expected_left_bx, 'left', car_center_bev, skip=right_cand)

                    # Initialization fallback: only for a side with NO expected position
                    if left_cand is None and _expected_left_bx is None:
                        for c in cand_list:
                            if c is right_cand:
                                continue
                            if c["bx"] < car_center_bev and (left_cand is None or
                                (car_center_bev - c["bx"]) < (car_center_bev - left_cand["bx"])):
                                left_cand = c
                    if right_cand is None and _expected_right_bx is None:
                        for c in cand_list:
                            if c is left_cand:
                                continue
                            if c["bx"] >= car_center_bev and (right_cand is None or
                                (c["bx"] - car_center_bev) < (right_cand["bx"] - car_center_bev)):
                                right_cand = c

                    left_lane = left_cand["pts"] if left_cand else None
                    right_lane = right_cand["pts"] if right_cand else None

                    # Update expected positions (SMA blend with detection)
                    if left_lane is not None:
                        det_bx = float(left_lane[np.argmax(left_lane[:, 1]), 0])
                        _expected_left_bx = _expected_left_bx * 0.3 + det_bx * 0.7 if _expected_left_bx is not None else det_bx
                        _expected_left_missed = 0
                    else:
                        _expected_left_missed += 1
                        if _expected_left_missed > 60:
                            _expected_left_bx = None
                            _left_history.clear()

                    if right_lane is not None:
                        det_bx = float(right_lane[np.argmax(right_lane[:, 1]), 0])
                        _expected_right_bx = _expected_right_bx * 0.3 + det_bx * 0.7 if _expected_right_bx is not None else det_bx
                        _expected_right_missed = 0
                    else:
                        _expected_right_missed += 1
                        if _expected_right_missed > 60:
                            _expected_right_bx = None
                            _right_history.clear()

                    # Update expected lane width
                    if left_lane is not None and right_lane is not None:
                        new_width = _expected_right_bx - _expected_left_bx
                        if 20.0 < new_width < 350.0:
                            if _expected_width_px is not None:
                                _expected_width_px = _expected_width_px * 0.5 + new_width * 0.5
                            else:
                                _expected_width_px = new_width

                    if left_lane is not None or right_lane is not None:
                        _intersection_hold = 0

                    for lane, label in [(left_lane, 'left'), (right_lane, 'right')]:
                        if lane is not None and len(lane) >= 4:
                            road = _bev_px_to_road(lane, w_bev, h_bev)
                            valid = ~np.isnan(road[:, 0]) & ~np.isnan(road[:, 1])
                            road = road[valid]
                            if len(road) >= 4:
                                poly = _robust_polyfit(road[:, 0], road[:, 1], 2)
                                if poly is not None:
                                    if max_a is not None and abs(poly[0]) > max_a:
                                        poly = None
                                if poly is not None:
                                    if label == 'left':
                                        left_coeffs = poly
                                    else:
                                        right_coeffs = poly

                    # ── Raw detection history ──
                    if left_coeffs is not None:
                        _left_history.append(left_coeffs.copy())
                        if len(_left_history) > _HISTORY_MAX:
                            _left_history.pop(0)
                    if right_coeffs is not None:
                        _right_history.append(right_coeffs.copy())
                        if len(_right_history) > _HISTORY_MAX:
                            _right_history.pop(0)

                    # ── Loss prediction ──
                    # Priority: own history → width-copy from other lane
                    if left_coeffs is None and right_coeffs is not None:
                        if _left_history:
                            left_coeffs = np.mean(_left_history[-5:], axis=0).copy()
                        elif _expected_width_px is not None:
                            rw = _expected_width_px * (2 * BEV_X_RANGE[1]) / BEV_DISPLAY_WIDTH
                            left_coeffs = right_coeffs.copy()
                            left_coeffs[2] = right_coeffs[2] - rw
                    if right_coeffs is None and left_coeffs is not None:
                        if _right_history:
                            right_coeffs = np.mean(_right_history[-5:], axis=0).copy()
                        elif _expected_width_px is not None:
                            rw = _expected_width_px * (2 * BEV_X_RANGE[1]) / BEV_DISPLAY_WIDTH
                            right_coeffs = left_coeffs.copy()
                            right_coeffs[2] = left_coeffs[2] + rw

                    # ── Both-lost recovery ──
                    # Priority: history average → intersection hold (vertical lines)
                    if left_coeffs is None and right_coeffs is None:
                        recovered = False
                        if _left_history and _right_history:
                            left_coeffs = np.mean(_left_history[-5:], axis=0).copy()
                            right_coeffs = np.mean(_right_history[-5:], axis=0).copy()
                            recovered = True
                        if not recovered:
                            should_hold = (_expected_left_bx is not None and _expected_right_bx is not None
                                           and _expected_width_px is not None
                                           and 20.0 < _expected_width_px < 350.0
                                           and _intersection_hold < 60)
                            if should_hold:
                                n_synth = 30
                                y_s = np.linspace(0, h_bev - 1, n_synth, dtype=np.float32)
                                left_bev = np.column_stack([np.full_like(y_s, _expected_left_bx), y_s])
                                right_bev = np.column_stack([np.full_like(y_s, _expected_right_bx), y_s])
                                lr = _bev_px_to_road(left_bev, w_bev, h_bev)
                                rr = _bev_px_to_road(right_bev, w_bev, h_bev)
                                lv = ~np.isnan(lr[:, 0]) & ~np.isnan(lr[:, 1])
                                rv = ~np.isnan(rr[:, 0]) & ~np.isnan(rr[:, 1])
                                if np.sum(lv) >= 4:
                                    left_coeffs = np.polyfit(lr[lv, 1], lr[lv, 0], 2)
                                if np.sum(rv) >= 4:
                                    right_coeffs = np.polyfit(rr[rv, 1], rr[rv, 0], 2)
                                _intersection_hold += 1

                    if left_coeffs is not None and right_coeffs is not None:
                        y_eval = 5.0
                        lx = np.polyval(left_coeffs, y_eval)
                        rx = np.polyval(right_coeffs, y_eval)
                        lw = rx - lx
                        if 1.0 < lw < 20.0:
                            lane_width = lw
                            lane_position = float(np.clip((0.0 - lx) / lw, 0.0, 1.0))
                else:
                    left_coeffs, right_coeffs = _fallback_fit_lanes(left_lane_pts, right_lane_pts, car_center_x)
                    if left_coeffs is not None and right_coeffs is not None:
                        y_eval = 5.0
                        lx = np.polyval(left_coeffs, y_eval)
                        rx = np.polyval(right_coeffs, y_eval)
                        lw = rx - lx
                        if lw > 0.5:
                            lane_width = lw
                            lane_position = float(np.clip((0.0 - lx) / lw, 0.0, 1.0))

                # Curvature consistency: reject lanes that curve in opposite directions
                if left_coeffs is not None and right_coeffs is not None:
                    l_curve = left_coeffs[0]
                    r_curve = right_coeffs[0]
                    if l_curve * r_curve < 0 and abs(l_curve) > 0.001 and abs(r_curve) > 0.001:
                        left_coeffs = None
                        right_coeffs = None

                # ── Curvature-aware smoothing ──
                # Normal EMA when stable; snap (alpha=1.0) when curve changes dramatically
                # so intersections and sharp transitions track immediately.
                base_alpha = 0.10 if motion.is_moving() else 0.35
                if left_coeffs is not None:
                    if _left_smooth is None or len(_left_smooth) != len(left_coeffs):
                        _left_smooth = left_coeffs.copy()
                    else:
                        curve_delta = abs(left_coeffs[0] - _left_smooth[0])
                        alpha = 1.0 if curve_delta > _SNAP_CURVE_THRESHOLD else base_alpha
                        _left_smooth = _left_smooth * (1 - alpha) + left_coeffs * alpha
                    _left_stale = 0
                elif _left_smooth is not None:
                    _left_stale += 1
                    if _left_stale > 30:
                        _left_smooth = None
                left_coeffs = _left_smooth

                if right_coeffs is not None:
                    if _right_smooth is None or len(_right_smooth) != len(right_coeffs):
                        _right_smooth = right_coeffs.copy()
                    else:
                        curve_delta = abs(right_coeffs[0] - _right_smooth[0])
                        alpha = 1.0 if curve_delta > _SNAP_CURVE_THRESHOLD else base_alpha
                        _right_smooth = _right_smooth * (1 - alpha) + right_coeffs * alpha
                    _right_stale = 0
                elif _right_smooth is not None:
                    _right_stale += 1
                    if _right_stale > 30:
                        _right_smooth = None
                right_coeffs = _right_smooth

                # Recompute lane width/position from smoothed coeffs
                if left_coeffs is not None and right_coeffs is not None:
                    y_eval = 5.0
                    lx = np.polyval(left_coeffs, y_eval)
                    rx = np.polyval(right_coeffs, y_eval)
                    lw = rx - lx
                    if 1.0 < lw < 20.0:
                        lane_width = lw
                        lane_position = float(np.clip((0.0 - lx) / lw, 0.0, 1.0))
                    else:
                        lane_width = 0.0
                        lane_position = 0.5

                result = {
                    "left_coeffs": left_coeffs, "right_coeffs": right_coeffs,
                    "lane_position": lane_position, "lane_width": lane_width,
                    "compensated_x": 0.0, "ego_center": None, "heading_error": 0.0,
                }
                left_coeffs = result["left_coeffs"]
                right_coeffs = result["right_coeffs"]
                lane_position = result["lane_position"]
                lane_width = result["lane_width"]
                compensated_x = result["compensated_x"]
                timer.track("lane_post", time.perf_counter() - _t_lane)

                _t_fcw = time.perf_counter()
                for det in detections:
                    cx = det["center_x"]
                    cy = det["center_y"]
                    is_threat = False
                    in_lane = False
                    if left_coeffs is not None and right_coeffs is not None:
                        left_x = np.polyval(left_coeffs, cy)
                        right_x = np.polyval(right_coeffs, cy)
                        in_lane = left_x <= cx <= right_x
                    else:
                        in_lane = abs(cx) < LANE_HALF_WIDTH
                    prev_cy = det.get("prev_cy", cy)
                    getting_closer = (prev_cy - cy) > 0.3
                    if in_lane and getting_closer and cy < FCW_WARNING_DISTANCE:
                        fcw_triggered = True
                        is_threat = True
                    det["threat"] = is_threat
                    fcw_overlay_boxes.append({
                        "x1": cx - det["width"] / 2.0,
                        "y1": cy + det["length"] / 2.0,
                        "x2": cx + det["width"] / 2.0,
                        "y2": cy - det["length"] / 2.0,
                        "threat": is_threat,
                    })
                timer.track("fcw", time.perf_counter() - _t_fcw)

                # LDW severity based on lane_position and calibrated comfort zones
                if lane_width > 0.5 and lane_position >= 0:
                    ldw_cal = state.get("ldw_calibration") or {}
                    left_pos = ldw_cal.get("left_pos", 0.25)
                    right_pos = ldw_cal.get("right_pos", 0.75)

                    d_left = lane_position
                    if d_left < left_pos and left_pos > 0:
                        left_severity = float(np.clip((left_pos - d_left) / left_pos, 0.0, 1.0))

                    d_right = 1.0 - lane_position
                    if d_right < (1.0 - right_pos) and right_pos < 1.0:
                        right_severity = float(np.clip(((1.0 - right_pos) - d_right) / (1.0 - right_pos), 0.0, 1.0))

                    if state["adas_enabled"]:
                        if left_severity > 0.8 or right_severity > 0.8:
                            ldw_triggered = True

                # LDW calibration data collection (records lane_position at edge of lane)
                _t_now = time.time()
                _cal_left = state.get("calibration_ldw_left_start_time")
                if _cal_left is not None and lane_width > 0.5 and lane_position >= 0:
                    if _t_now - _cal_left < LDW_CALIBRATION_SECONDS:
                        state.setdefault("calib_ldw_left_positions", []).append(lane_position)
                    else:
                        _ps = state.get("calib_ldw_left_positions", [])
                        if _ps:
                            _med = float(np.median(_ps))
                            _c = state.get("ldw_calibration") or {}
                            _c["left_pos"] = _med
                            state["ldw_calibration"] = _c
                            print(f"[LDW] left cal done: lane_pos={_med:.3f}", flush=True)
                            save_calibration_state()
                        state["calibration_ldw_left_start_time"] = None

                _cal_right = state.get("calibration_ldw_right_start_time")
                if _cal_right is not None and lane_width > 0.5 and lane_position >= 0:
                    if _t_now - _cal_right < LDW_CALIBRATION_SECONDS:
                        state.setdefault("calib_ldw_right_positions", []).append(lane_position)
                    else:
                        _ps = state.get("calib_ldw_right_positions", [])
                        if _ps:
                            _med = float(np.median(_ps))
                            _c = state.get("ldw_calibration") or {}
                            _c["right_pos"] = _med
                            state["ldw_calibration"] = _c
                            print(f"[LDW] right cal done: lane_pos={_med:.3f}", flush=True)
                            save_calibration_state()
                        state["calibration_ldw_right_start_time"] = None

            # ── Lead car detection (before rendering, so lane area can use its color) ──
            lead_car = None
            lead_car_dist = None
            lead_car_color = None
            if detections:
                for det in detections:
                    cy = det["center_y"]
                    if cy <= 0:
                        continue
                    in_lane = False
                    if _bev_warp_M is not None and left_coeffs is not None and right_coeffs is not None:
                        cx_img = (det.get("img_x1", 0) + det.get("img_x2", 0)) / 2.0
                        by_img = det.get("img_y2", 0)
                        pt_img = np.array([[[cx_img, by_img]]], dtype=np.float32)
                        pt_bev = cv2.perspectiveTransform(pt_img, _bev_warp_M)[0][0]
                        bvx, bvy = pt_bev[0], pt_bev[1]
                        if 0 <= bvy < BEV_DISPLAY_HEIGHT:
                            ry = (1.0 - bvy / BEV_DISPLAY_HEIGHT) * BEV_Y_RANGE[1]
                            ry = np.clip(ry, 1.5, 18.0)
                            left_rx = np.polyval(left_coeffs, ry)
                            right_rx = np.polyval(right_coeffs, ry)
                            left_bvx = ((left_rx / (2 * BEV_X_RANGE[1]) + 0.5) * BEV_DISPLAY_WIDTH)
                            right_bvx = ((right_rx / (2 * BEV_X_RANGE[1]) + 0.5) * BEV_DISPLAY_WIDTH)
                            in_lane = left_bvx <= bvx <= right_bvx
                        if in_lane:
                            inv_bev = np.linalg.inv(_bev_warp_M)
                            lane_bev = np.array([[[left_bvx, bvy], [right_bvx, bvy]]], dtype=np.float32)
                            lane_img = cv2.perspectiveTransform(lane_bev, inv_bev)[0]
                            lix, rix = lane_img[0, 0], lane_img[1, 0]
                            margin = (rix - lix) * 0.2
                            if cx_img < lix - margin or cx_img > rix + margin:
                                in_lane = False
                            edge_margin = CAPTURE_WIDTH * 0.10
                            if cx_img < edge_margin or cx_img > CAPTURE_WIDTH - edge_margin:
                                in_lane = False
                    elif left_coeffs is not None and right_coeffs is not None:
                        cx = det["center_x"]
                        left_x = np.polyval(left_coeffs, cy)
                        right_x = np.polyval(right_coeffs, cy)
                        in_lane = left_x <= cx <= right_x
                    else:
                        in_lane = abs(det["center_x"]) < LANE_HALF_WIDTH
                    if in_lane:
                        dyc = state.get("drivable_y_cutoff")
                        if dyc is not None and det.get("img_y2", CAPTURE_HEIGHT) > dyc:
                            in_lane = False
                    if in_lane and (lead_car is None or cy < lead_car["center_y"]):
                        lead_car = det

            if lead_car is not None:
                if _bev_warp_M is not None:
                    cx_img = (lead_car.get("img_x1", 0) + lead_car.get("img_x2", 0)) / 2.0
                    by_img = lead_car.get("img_y2", 0)
                    pt_img = np.array([[[cx_img, by_img]]], dtype=np.float32)
                    pt_bev = cv2.perspectiveTransform(pt_img, _bev_warp_M)[0][0]
                    lead_car_dist = (1.0 - pt_bev[1] / BEV_DISPLAY_HEIGHT) * BEV_Y_RANGE[1]
                else:
                    lead_car_dist = lead_car.get("center_y", 20.0)
                lead_car_dist = max(lead_car_dist, 0.5)
                t = np.clip((lead_car_dist - 5.0) / (20.0 - 5.0), 0.0, 1.0)
                if t < 0.5:
                    ri, gi, bi = 255, int(255 * t * 2), 0
                else:
                    ri, gi, bi = int(255 * (1 - (t - 0.5) * 2)), 255, 0
                lead_car_color = f"#{ri:02x}{gi:02x}{bi:02x}"

            # ── Rendering ───────────────────────────────────────────────────
            _t_render = time.perf_counter()
            if _bev_warp_M is not None:
                im_bev = cv2.warpPerspective(
                    im0, _bev_warp_M,
                    (BEV_DISPLAY_WIDTH, BEV_DISPLAY_HEIGHT),
                    flags=cv2.INTER_LINEAR,
                )
                im_bev = render_bev_frame(
                    lane_mask, detections,
                    left_coeffs, right_coeffs,
                    compensated_x, lane_position, lane_width,
                    fcw_triggered, left_severity, right_severity,
                    bg_img=im_bev, bev_debug=bev_debug, raw_lanes=lanes_bev,
                )
                _draw_filtered_lanes_perspective(im_debug, left_coeffs, right_coeffs, _bev_warp_M, lane_position, lead_car_color)
            else:
                im_bev = render_bev_frame(
                    lane_mask, detections,
                    left_coeffs, right_coeffs,
                    compensated_x, lane_position, lane_width,
                    fcw_triggered, left_severity, right_severity,
                )
                if left_coeffs is not None or right_coeffs is not None:
                    h_im, w_im = im_debug.shape[:2]
                    y_vals = np.arange(2.0, 18.0, 0.5, dtype=np.float32)
                    left_img = right_img = None
                    if left_coeffs is not None:
                        road_x = np.polyval(left_coeffs, y_vals)
                        img_pts = _road_to_image(np.column_stack([road_x, y_vals]))
                        if img_pts is not None:
                            valid = ~np.isnan(img_pts[:, 0])
                            left_img = img_pts[valid].astype(np.int32)
                    if right_coeffs is not None:
                        road_x = np.polyval(right_coeffs, y_vals)
                        img_pts = _road_to_image(np.column_stack([road_x, y_vals]))
                        if img_pts is not None:
                            valid = ~np.isnan(img_pts[:, 0])
                            right_img = img_pts[valid].astype(np.int32)

                    # Lane area fill
                    if left_img is not None and right_img is not None and len(left_img) >= 2 and len(right_img) >= 2:
                        min_len = min(len(left_img), len(right_img))
                        lp = left_img[:min_len].copy()
                        rp = right_img[:min_len].copy()
                        lp[:, 0] = (lp[:, 0] * w_im / CAPTURE_WIDTH).astype(np.int32)
                        lp[:, 1] = (lp[:, 1] * h_im / CAPTURE_HEIGHT).astype(np.int32)
                        rp[:, 0] = (rp[:, 0] * w_im / CAPTURE_WIDTH).astype(np.int32)
                        rp[:, 1] = (rp[:, 1] * h_im / CAPTURE_HEIGHT).astype(np.int32)
                        poly = np.vstack([lp, rp[::-1]])
                        if lead_car_color:
                            area_color = (int(lead_car_color[5:7], 16), int(lead_car_color[3:5], 16), int(lead_car_color[1:3], 16))
                        else:
                            area_color = (255, 100, 0)
                        ov = im_debug.copy()
                        cv2.fillPoly(ov, [poly], area_color)
                        cv2.addWeighted(ov, 0.25, im_debug, 0.75, 0, im_debug)

                    # Gradient lane lines
                    lp_clamped = max(0.0, min(1.0, lane_position))
                    left_shift = 2 * abs(lp_clamped - 0.5) if lp_clamped < 0.5 else 0.0
                    right_shift = 2 * abs(lp_clamped - 0.5) if lp_clamped > 0.5 else 0.0
                    for pts, shift in [(left_img, left_shift), (right_img, right_shift)]:
                        if pts is None or len(pts) < 2:
                            continue
                        p = pts.copy()
                        p[:, 0] = (p[:, 0] * w_im / CAPTURE_WIDTH).astype(np.int32)
                        p[:, 1] = (p[:, 1] * h_im / CAPTURE_HEIGHT).astype(np.int32)
                        b = int(255 * (1 - shift))
                        g = int(100 * (1 - shift))
                        r = int(255 * shift)
                        for i in range(len(p) - 1):
                            cv2.line(im_debug, tuple(p[i]), tuple(p[i + 1]), (b, g, r), 3, cv2.LINE_AA)

            # ── Lead car mask rendering ─────────────────────────────────────
            if lead_car is not None:
                sx = im_debug.shape[1] / CAPTURE_WIDTH
                sy = im_debug.shape[0] / CAPTURE_HEIGHT
                x1 = int(lead_car.get("img_x1", 0) * sx)
                y1 = int(lead_car.get("img_y1", 0) * sy)
                x2 = int(lead_car.get("img_x2", 0) * sx)
                y2 = int(lead_car.get("img_y2", 0) * sy)
                r = int(lead_car_color[1:3], 16)
                g = int(lead_car_color[3:5], 16)
                b = int(lead_car_color[5:7], 16)
                overlay = im_debug.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)
                cv2.addWeighted(overlay, 0.35, im_debug, 0.65, 0, im_debug)
                cv2.rectangle(im_debug, (x1, y1), (x2, y2), (b, g, r), 2, cv2.LINE_AA)

            # ── Frontend overlay data ──────────────────────────────────────
            global _perspective_overlay_data
            overlay_lanes = []
            overlay_lane_fill = None
            if _bev_warp_M is not None:
                bw, bh = BEV_DISPLAY_WIDTH, BEV_DISPLAY_HEIGHT
                xr, yr = BEV_X_RANGE[1], BEV_Y_RANGE[1]
                y_vals = np.arange(2.0, 18.0, 0.5, dtype=np.float32)
                sx, sy = im_debug.shape[1] / CAPTURE_WIDTH, im_debug.shape[0] / CAPTURE_HEIGHT
                S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
                M_inv = S @ np.linalg.inv(_bev_warp_M)

                left_pts = right_pts = None
                if left_coeffs is not None:
                    rx = np.polyval(left_coeffs, y_vals)
                    u = ((rx / (2 * xr) + 0.5) * bw).astype(np.float32)
                    v = ((1.0 - y_vals / yr) * bh).astype(np.float32)
                    pts_bev = np.column_stack([u, v]).reshape(1, -1, 2).astype(np.float32)
                    pts_cam = cv2.perspectiveTransform(pts_bev, M_inv).reshape(-1, 2).astype(np.float32)
                    overlay_lanes.append([float(v) for pt in pts_cam for v in pt])
                    left_pts = pts_cam
                if right_coeffs is not None:
                    rx = np.polyval(right_coeffs, y_vals)
                    u = ((rx / (2 * xr) + 0.5) * bw).astype(np.float32)
                    v = ((1.0 - y_vals / yr) * bh).astype(np.float32)
                    pts_bev = np.column_stack([u, v]).reshape(1, -1, 2).astype(np.float32)
                    pts_cam = cv2.perspectiveTransform(pts_bev, M_inv).reshape(-1, 2).astype(np.float32)
                    overlay_lanes.append([float(v) for pt in pts_cam for v in pt])
                    right_pts = pts_cam

                if left_pts is not None and right_pts is not None:
                    min_len = min(len(left_pts), len(right_pts))
                    poly = np.vstack([left_pts[:min_len], right_pts[:min_len][::-1]])
                    overlay_lane_fill = {
                        "points": [float(v) for pt in poly for v in pt],
                        "color": lead_car_color if lead_car_color else "#0064ff",
                    }

            overlay_vehicles = []
            if lead_car is not None:
                sx = im_debug.shape[1] / CAPTURE_WIDTH
                sy = im_debug.shape[0] / CAPTURE_HEIGHT
                overlay_vehicles.append({
                    "x1": float(lead_car.get("img_x1", 0) * sx),
                    "y1": float(lead_car.get("img_y1", 0) * sy),
                    "x2": float(lead_car.get("img_x2", 0) * sx),
                    "y2": float(lead_car.get("img_y2", 0) * sy),
                    "id": lead_car.get("track_id", -1),
                    "distance": lead_car_dist,
                    "color": lead_car_color,
                })
            _perspective_overlay_data = {
                "width": im_debug.shape[1],
                "height": im_debug.shape[0],
                "lanes": overlay_lanes,
                "lane_fill": overlay_lane_fill,
                "vehicles": overlay_vehicles,
                "lane_position": lane_position,
            }

            state["moving"] = motion.is_moving()
            state["left_severity"] = left_severity
            state["right_severity"] = right_severity
            state["lane_width"] = lane_width
            timer.track("bev_render", time.perf_counter() - _t_render)

            state["fcw_warning"] = fcw_triggered
            state["ldw_warning"] = ldw_triggered

            _t_enc = time.perf_counter()
            _, buf = cv2.imencode('.jpg', im_bev, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if state["debug_feed_active"]:
                _, buf_debug = cv2.imencode('.jpg', im_debug, [cv2.IMWRITE_JPEG_QUALITY, 85])
            timer.track("encode", time.perf_counter() - _t_enc)

            with frame_lock:
                latest_web_frame = buf.tobytes()
                if state["debug_feed_active"]:
                    latest_debug_frame = buf_debug.tobytes()

            state["last_frame_time"] = time.time()
            timer.track("total", time.perf_counter() - _t_frame)
            timer.maybe_log()

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                state["web_fps"] = round(fps_counter / elapsed, 1)
                fps_counter = 0
                fps_start = time.time()

        except Exception:
            logging.exception("inference_loop failed")
            time.sleep(0.05)


def generate_mjpeg():
    try:
        while True:
            with frame_lock:
                frame = latest_web_frame
            if frame is None:
                wait_img = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(wait_img, "Initializing...",
                            (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                _, buf = cv2.imencode('.jpg', wait_img)
                frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    except GeneratorExit:
        pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_debug_mjpeg():
    no_feed_frame = None
    try:
        while True:
            with frame_lock:
                is_active = state.get("debug_feed_active", True)
                frame = latest_debug_frame
            if not is_active:
                if no_feed_frame is None:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "Perspective feed disabled",
                                (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (100, 100, 100), 2)
                    _, buf = cv2.imencode('.jpg', img)
                    no_feed_frame = buf.tobytes()
                frame = no_feed_frame
            elif frame is None:
                wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(wait_img, "Initializing Debug Feed...",
                            (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                _, buf = cv2.imencode('.jpg', wait_img)
                frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    except GeneratorExit:
        pass


@app.route('/debug_feed')
def debug_feed():
    return Response(generate_debug_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/lane_overlay')
def lane_overlay():
    with frame_lock:
        return jsonify({
            "width": BEV_DISPLAY_WIDTH,
            "height": BEV_DISPLAY_HEIGHT,
            "top_y": 0,
            "bottom_y": BEV_DISPLAY_HEIGHT - 1,
            "left_points": [],
            "right_points": [],
            "polygon": [],
        })


@app.route('/api/perspective_overlay')
def perspective_overlay():
    return jsonify(_perspective_overlay_data)


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

    calibrating_ldw_right = False
    calibration_ldw_right_progress = 0
    if state.get("calibration_ldw_right_start_time") is not None:
        elapsed = now - state["calibration_ldw_right_start_time"]
        if elapsed < LDW_CALIBRATION_SECONDS:
            calibrating_ldw_right = True
            calibration_ldw_right_progress = int((elapsed / LDW_CALIBRATION_SECONDS) * 100)

    return jsonify({
        "capture_fps": state["capture_fps"],
        "web_fps": state["web_fps"],
        "recording": state["recording"],
        "adas_enabled": state["adas_enabled"],
        "fcw_warning": state["fcw_warning"],
        "ldw_warning": state["ldw_warning"],
        "recording_duration": duration,
        "error": state["error"],
        "lane_position": state.get("lane_position", 0.5),
        "lane_width": state.get("lane_width", 0.0),
        "debug_feed_active": state.get("debug_feed_active", True),
        "seg_classes": [],
        "trace_start_y": 370,
        "calibrating_lens": False,
        "calibration_lens_progress": 100,
        "calibrating_ldw_left": calibrating_ldw_left,
        "calibration_ldw_left_progress": calibration_ldw_left_progress,
        "calibrating_ldw_right": calibrating_ldw_right,
        "calibration_ldw_right_progress": calibration_ldw_right_progress,
        "calibration_total": 100,
        "lens_calibrated": state["calibration"] is not None,
        "ldw_calibrated": state["ldw_calibration"] is not None,
        "ldw_calibrated_left": state["ldw_calibration"] is not None and "left_pos" in state["ldw_calibration"],
        "ldw_calibrated_right": state["ldw_calibration"] is not None and "right_pos" in state["ldw_calibration"],
        "left_severity": state.get("left_severity", 0.0),
        "right_severity": state.get("right_severity", 0.0),
        "car_center_x": state.get("car_center_x", 640),
        "moving": state.get("moving", False),
        "cuda_available": True,
        "gpu_device_name": "NVIDIA Jetson",
        "model": "YOLOPv2",
    })


@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    data = request.json
    state["recording"] = data.get("enabled", not state["recording"])
    return jsonify({"success": True, "recording": state["recording"]})


@app.route('/api/toggle_adas', methods=['POST'])
def api_toggle_adas():
    data = request.json
    state["adas_enabled"] = data.get('enabled', False)
    return jsonify({"adas_enabled": state["adas_enabled"]})


@app.route('/api/toggle_debug_feed', methods=['POST'])
def api_toggle_debug_feed():
    data = request.json
    state["debug_feed_active"] = data.get('enabled', not state["debug_feed_active"])
    return jsonify({"debug_feed_active": state["debug_feed_active"]})


@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    state["calibrate_requested"] = True
    state["calib_vp_x_list"] = []
    state["calib_vp_y_list"] = []
    state["calibration_frames_left"] = 0
    state["error"] = None
    return jsonify({"status": "calibrating"})


@app.route('/api/calibrate_ldw_left', methods=['POST'])
def api_calibrate_ldw_left():
    state["calibration_ldw_left_start_time"] = time.time()
    state["calib_ldw_left_positions"] = []
    state["calibration_ldw_right_start_time"] = None
    state["error"] = None
    return jsonify({"status": "calibrating_left"})


@app.route('/api/calibrate_ldw_right', methods=['POST'])
def api_calibrate_ldw_right():
    state["calibration_ldw_right_start_time"] = time.time()
    state["calib_ldw_right_positions"] = []
    state["calibration_ldw_left_start_time"] = None
    state["error"] = None
    return jsonify({"status": "calibrating_right"})


@app.route('/api/set_center_x', methods=['POST'])
def api_set_center_x():
    data = request.json or {}
    cx = data.get('car_center_x')
    if cx is None:
        return jsonify({"success": False, "error": "car_center_x required"}), 400
    state["car_center_x"] = int(np.clip(int(cx), 0, 1279))
    save_calibration_state()
    return jsonify({"success": True, "car_center_x": state["car_center_x"]})


@app.route('/api/set_trace_start', methods=['POST'])
def api_set_trace_start():
    return jsonify({"success": True})


@app.route('/api/set_crop', methods=['POST'])
def api_set_crop():
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# BEV calibration tool
# ---------------------------------------------------------------------------

bev_cal = {
    "active": False,
    "src_points": [],  # 4 [x,y] in source image (1280x800) space
    "dst_points": [],  # 4 [x,y] in BEV output space
    "bev_width": 500,
    "bev_height": 500,
}

DEFAULT_BEV_DST = [[20, 480], [480, 480], [480, 20], [20, 20]]


@app.route('/api/bev_cal/start', methods=['POST'])
def api_bev_cal_start():
    bev_cal["active"] = True
    bev_cal["src_points"] = []
    bev_cal["dst_points"] = [list(p) for p in DEFAULT_BEV_DST]
    return jsonify(bev_cal)


@app.route('/api/bev_cal/clear', methods=['POST'])
def api_bev_cal_clear():
    bev_cal["active"] = False
    bev_cal["src_points"] = []
    bev_cal["dst_points"] = []
    return jsonify(bev_cal)


@app.route('/api/bev_cal/set_src', methods=['POST'])
def api_bev_cal_set_src():
    data = request.json
    idx = int(data["index"])
    while len(bev_cal["src_points"]) <= idx:
        bev_cal["src_points"].append(None)
    bev_cal["src_points"][idx] = [int(data["x"]), int(data["y"])]
    return jsonify(bev_cal)


@app.route('/api/bev_cal/set_dst', methods=['POST'])
def api_bev_cal_set_dst():
    data = request.json
    idx = int(data["index"])
    while len(bev_cal["dst_points"]) <= idx:
        bev_cal["dst_points"].append(None)
    bev_cal["dst_points"][idx] = [int(data["x"]), int(data["y"])]
    return jsonify(bev_cal)


@app.route('/api/bev_cal/state')
def api_bev_cal_state():
    return jsonify(bev_cal)


@app.route('/api/bev_cal/frame')
def api_bev_cal_frame():
    global calibration_frame
    frame = calibration_frame
    if frame is None:
        return jsonify({"error": "no frame"}), 400
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/bev_cal/preview')
def api_bev_cal_preview():
    if not bev_cal["active"]:
        return jsonify({"error": "not active"}), 400
    src = bev_cal["src_points"]
    dst = bev_cal["dst_points"]
    if len(src) < 4 or any(p is None for p in src):
        return jsonify({"error": "need 4 source points"}), 400
    if len(dst) < 4 or any(p is None for p in dst):
        return jsonify({"error": "need 4 destination points"}), 400

    global calibration_frame
    frame = calibration_frame
    if frame is None:
        return jsonify({"error": "no frame"}), 400

    M = cv2.getPerspectiveTransform(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
    )
    warped = cv2.warpPerspective(frame, M, (bev_cal["bev_width"], bev_cal["bev_height"]))
    _, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/bev_cal/save', methods=['POST'])
def api_bev_cal_save():
    src = bev_cal["src_points"]
    dst = bev_cal["dst_points"]
    if len(src) < 4 or any(p is None for p in src):
        return jsonify({"success": False, "error": "need 4 source points"}), 400
    if len(dst) < 4 or any(p is None for p in dst):
        return jsonify({"success": False, "error": "need 4 destination points"}), 400

    calib = {}
    calib_path = "models/calibration.json"
    if os.path.exists(calib_path):
        try:
            with open(calib_path) as f:
                calib = json.load(f)
        except Exception:
            pass
    calib["bev_cal_src"] = src
    calib["bev_cal_dst"] = dst
    calib["bev_cal_active"] = True
    os.makedirs("models", exist_ok=True)
    with open(calib_path, "w") as f:
        json.dump(calib, f)
    return jsonify({"success": True, "message": "BEV calibration saved"})


def _load_bev_calibration():
    global _bev_warp_M
    calib_path = "models/calibration.json"
    if not os.path.exists(calib_path):
        return
    try:
        with open(calib_path) as f:
            data = json.load(f)
        if "bev_cal_src" in data and "bev_cal_dst" in data:
            bev_cal["src_points"] = data["bev_cal_src"]
            bev_cal["dst_points"] = data["bev_cal_dst"]
            bev_cal["active"] = data.get("bev_cal_active", True)
            src = bev_cal["src_points"]
            dst = bev_cal["dst_points"]
            if len(src) == 4 and len(dst) == 4 and all(p is not None for p in src) and all(p is not None for p in dst):
                _bev_warp_M = cv2.getPerspectiveTransform(
                    np.array(src, dtype=np.float32),
                    np.array(dst, dtype=np.float32),
                )
            print("[INFO] Loaded BEV calibration", flush=True)
    except Exception as e:
        print(f"[WARN] BEV calibration load failed: {e}", flush=True)


_load_bev_calibration()


def _watchdog_loop():
    while True:
        time.sleep(5.0)
        now = time.time()
        inf_time = state.get("last_frame_time", 0.0)
        cap_time = state.get("capture_last_frame_time", 0.0)
        if inf_time > 0 and now - inf_time > 8.0:
            print(f"[WATCHDOG] Inference thread stalled for {now - inf_time:.0f}s "
                  f"(capture last frame: {now - cap_time:.0f}s ago)", flush=True)
        if cap_time > 0 and now - cap_time > 8.0:
            print(f"[WATCHDOG] Capture thread stalled for {now - cap_time:.0f}s "
                  f"(inference last frame: {now - inf_time:.0f}s ago)", flush=True)


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("Edge ADAS — YOLOPv2 + homography lane projection", flush=True)
    print("=" * 60, flush=True)

    t_watchdog = threading.Thread(target=_watchdog_loop, daemon=True)
    t_watchdog.start()

    # Build TensorRT engine before starting anything else
    TRT_PRECISION = os.environ.get("TRT_PRECISION", "fp16")
    TRT_ENGINE_PATH = f"data/weights/yolopv2_{TRT_PRECISION}.engine"
    import build_engines
    trt_engine = build_engines.ensure_yolop_trt_engine(
        model_path="data/weights/yolopv2.pt",
        onnx_path="data/weights/yolopv2.onnx",
        engine_path=TRT_ENGINE_PATH,
        precision=TRT_PRECISION,
    )
    os.environ["YOLOP_TRT_ENGINE"] = trt_engine
    print(f"[INFO] TensorRT engine path: {trt_engine}", flush=True)

    print("Starting Capture thread...", flush=True)
    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_capture.start()

    print("Starting Inference thread...", flush=True)
    t_inference = threading.Thread(target=inference_loop, daemon=True)
    t_inference.start()

    print("Starting Flask server on port 5001...", flush=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
