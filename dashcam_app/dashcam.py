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


def compute_severity(distance, comfort_dist=1.1, threshold=0.3):
    if distance is None or distance > comfort_dist:
        return 0.0
    severity = (comfort_dist - distance) / threshold
    return float(np.clip(severity, 0.0, 1.0))

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
            col_sums = np.sum(strip, axis=0)
            if np.max(col_sums) == 0:
                wins.append((x1, y1, x2, y2, None, None))
                break
            best_col = np.argmax(col_sums)
            mx = x1 + float(best_col)
            if abs(mx - cx) > win_w * 0.6:
                wins.append((x1, y1, x2, y2, None, None))
                break
            ys_win, _ = np.where(strip > 0)
            my = y1 + float(np.mean(ys_win))
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

    if peaks:
        tracked_count = sum(1 for a in all_attempts if a["tracked"])
        print(f"[LANE] {len(peaks)} peaks, {tracked_count} lanes tracked "
              f"y_starts={[a['y_start'] for a in all_attempts]}", flush=True)
    else:
        print(f"[LANE] No peaks (thresh={thresh:.0f}, max_hist={np.max(hist_smooth):.0f})", flush=True)

    return lanes, {
        "bev_mask": bev_mask,
        "peaks": peaks,
        "attempts": all_attempts,
    }


latest_web_frame = None
latest_debug_frame = None
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
    "calib_ldw_left_dists": [],
    "calib_ldw_right_dists": [],
    "ldw_calibration": None,
    "car_center_x": 640,
    "car_center_offset": 0.0,
    "left_poly_history": [],
    "right_poly_history": [],
    "last_frame_time": 0.0,
    "capture_last_frame_time": 0.0,
}

video_writer = None

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
            self._cap = cv2.VideoCapture(VIDEO_SOURCE)
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
    global raw_frame_buffer, video_writer, calibration_frame

    cap = _CamWrapper()

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
                     bg_img=None, bev_debug=None):
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
                if attempt["tracked"] and len(attempt["lane_pts"]) >= 2:
                    pts_arr = np.array(attempt["lane_pts"], dtype=np.int32)
                    cv2.polylines(im_bev, [pts_arr], False, (0, 255, 255), 2)
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

        if len(left_pts) >= 2:
            cv2.polylines(im_bev, [np.array(left_pts, dtype=np.int32)],
                          False, (220, 220, 220), 2, cv2.LINE_AA)
        if len(right_pts) >= 2:
            cv2.polylines(im_bev, [np.array(right_pts, dtype=np.int32)],
                          False, (220, 220, 220), 2, cv2.LINE_AA)

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


def _draw_path_ribbon(im, left_coeffs, right_coeffs, lane_width):
    if left_coeffs is None or right_coeffs is None or lane_width < 1.0:
        return
    global _bev_warp_M
    if _bev_warp_M is None:
        return

    h_im, w_im = im.shape[:2]
    M_inv = np.linalg.inv(_bev_warp_M)

    bw = BEV_DISPLAY_WIDTH
    bh = BEV_DISPLAY_HEIGHT
    xr = BEV_X_RANGE[1]
    yr = BEV_Y_RANGE[1]

    y_vals = np.arange(1.0, 14.0, 0.5, dtype=np.float32)
    left_x = np.polyval(left_coeffs, y_vals)
    right_x = np.polyval(right_coeffs, y_vals)
    center_x = (left_x + right_x) / 2.0
    half_w = max(lane_width / 4.0, 0.5)

    def _road_to_bev_px(rx, ry):
        u = ((rx / (2 * xr) + 0.5) * bw).astype(np.float32)
        v = ((1.0 - ry / yr) * bh).astype(np.float32)
        return np.column_stack([u, v])

    pts_left = _road_to_bev_px(center_x - half_w, y_vals)
    pts_right = _road_to_bev_px(center_x + half_w, y_vals[::-1])
    pts_ribbon = np.vstack([pts_left, pts_right]).reshape(1, -1, 2)

    pts_cam = cv2.perspectiveTransform(pts_ribbon, M_inv).reshape(-1, 2).astype(np.int32)
    pts_cam[:, 0] = np.clip(pts_cam[:, 0], 0, w_im - 1)
    pts_cam[:, 1] = np.clip(pts_cam[:, 1], 0, h_im - 1)

    if len(pts_cam) >= 3:
        overlay = im.copy()
        cv2.fillPoly(overlay, [pts_cam], (0, 180, 255))
        cv2.addWeighted(overlay, 0.35, im, 0.65, 0, dst=im)

    pts_center = _road_to_bev_px(center_x, y_vals).reshape(1, -1, 2)
    pts_cen_cam = cv2.perspectiveTransform(pts_center, M_inv).reshape(-1, 2).astype(np.int32)
    if len(pts_cen_cam) >= 2:
        cv2.polylines(im, [pts_cen_cam], False, (255, 255, 255), 2, cv2.LINE_AA)


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


def inference_loop():
    global latest_web_frame, latest_debug_frame, raw_frame_buffer

    _left_smooth = None
    _right_smooth = None
    _left_stale = 0
    _right_stale = 0

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
                            })
                if ll_mask is not None:
                    lane_mask = ll_mask
                    da_mask_big = cv2.resize(da_mask, (im_debug.shape[1], im_debug.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    ll_mask_big = cv2.resize(ll_mask, (im_debug.shape[1], im_debug.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    da_overlay = np.zeros_like(im_debug)
                    da_overlay[da_mask_big > 0] = (40, 40, 40)
                    cv2.addWeighted(da_overlay, 0.4, im_debug, 0.6, 0, dst=im_debug)
                    car_center_x = state.get("car_center_x", INFER_WIDTH // 2)
                    draw_scale = np.array([im_debug.shape[1] / ll_mask.shape[1],
                                           im_debug.shape[0] / ll_mask.shape[0]], dtype=np.float32)
                    tr_start = int(ll_mask.shape[0] * TRACE_START_Y / 480)
                    left_edge, right_edge = _compute_lane_boundaries(da_mask, ll_mask, car_center_x, tr_start)
                    ll_overlay = np.zeros_like(im_debug)
                    ll_overlay[ll_mask_big > 0] = (0, 255, 255)
                    cv2.addWeighted(ll_overlay, 0.3, im_debug, 0.7, 0, dst=im_debug)
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

                if _bev_warp_M is not None and ll_mask is not None:
                    lanes_bev, bev_debug = _bev_detect_lanes(ll_mask, cam_shape=im0.shape[:2])
                    w_bev = BEV_DISPLAY_WIDTH
                    h_bev = BEV_DISPLAY_HEIGHT
                    car_center_bev = w_bev // 2

                    left_lane = None
                    right_lane = None
                    left_best = float('inf')
                    right_best = float('inf')
                    for pts in lanes_bev:
                        if len(pts) < 3:
                            continue
                        bot = pts[np.argmax(pts[:, 1])]
                        bx = bot[0]
                        if bx < car_center_bev:
                            d = car_center_bev - bx
                            if d < left_best:
                                left_best = d
                                left_lane = pts
                        else:
                            d = bx - car_center_bev
                            if d < right_best:
                                right_best = d
                                right_lane = pts

                    for lane, label in [(left_lane, 'left'), (right_lane, 'right')]:
                        if lane is not None and len(lane) >= 4:
                            road = _bev_px_to_road(lane, w_bev, h_bev)
                            valid = ~np.isnan(road[:, 0]) & ~np.isnan(road[:, 1])
                            road = road[valid]
                            if len(road) >= 4:
                                poly = _robust_polyfit(road[:, 0], road[:, 1], 2)
                                if poly is not None:
                                    if label == 'left':
                                        left_coeffs = poly
                                    else:
                                        right_coeffs = poly

                    if left_coeffs is not None and right_coeffs is not None:
                        y_eval = 5.0
                        lx = np.polyval(left_coeffs, y_eval)
                        rx = np.polyval(right_coeffs, y_eval)
                        lw = rx - lx
                        if 2.0 < lw < 5.0:
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

                # EMA smoothing
                alpha = 0.35
                if left_coeffs is not None:
                    if _left_smooth is None or len(_left_smooth) != len(left_coeffs):
                        _left_smooth = left_coeffs.copy()
                    else:
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
                    if 2.0 < lw < 5.0:
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
                    if abs(cx - compensated_x) < LANE_HALF_WIDTH and cy < FCW_WARNING_DISTANCE:
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

                ldw_cal = state.get("ldw_calibration") or {}
                lc = ldw_cal.get("left_comfort_dist", 1.1)
                rc = ldw_cal.get("right_comfort_dist", 1.1)
                if lc > 10.0:
                    lc = 1.1
                if rc > 10.0:
                    rc = 1.1

                y_eval = 5.0
                if left_coeffs is not None:
                    lx = np.polyval(left_coeffs, y_eval)
                    d_left = compensated_x - lx
                else:
                    d_left = 999.0
                if right_coeffs is not None:
                    rx = np.polyval(right_coeffs, y_eval)
                    d_right = rx - compensated_x
                else:
                    d_right = 999.0

                left_severity = compute_severity(d_left, lc)
                right_severity = compute_severity(d_right, rc)

                if state["adas_enabled"]:
                    if left_severity > 0.8 or right_severity > 0.8:
                        ldw_triggered = True

                _draw_path_ribbon(im_debug, left_coeffs, right_coeffs, lane_width)

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
                    bg_img=im_bev, bev_debug=bev_debug,
                )
            else:
                im_bev = render_bev_frame(
                    lane_mask, detections,
                    left_coeffs, right_coeffs,
                    compensated_x, lane_position, lane_width,
                    fcw_triggered, left_severity, right_severity,
                )

            state["lane_position"] = lane_position
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
        "ldw_calibrated_left": state["ldw_calibration"] is not None and "left_comfort_dist" in state["ldw_calibration"],
        "ldw_calibrated_right": state["ldw_calibration"] is not None and "right_comfort_dist" in state["ldw_calibration"],
        "car_center_x": state.get("car_center_x", 640),
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
    state["calib_ldw_left_dists"] = []
    state["calibration_ldw_right_start_time"] = None
    state["error"] = None
    return jsonify({"status": "calibrating_left"})


@app.route('/api/calibrate_ldw_right', methods=['POST'])
def api_calibrate_ldw_right():
    state["calibration_ldw_right_start_time"] = time.time()
    state["calib_ldw_right_dists"] = []
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
