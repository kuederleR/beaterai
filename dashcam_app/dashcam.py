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

from deepstream_pipeline import create_pipeline
from bevformer_detector import BEVFormerDetector, BEV_GRID_SIZE, BEV_X_RANGE, BEV_Y_RANGE
from adas_postprocessor import ADASPostprocessor, compute_severity

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



latest_web_frame = None
latest_debug_frame = None
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
    _, buf = cv2.imencode('.bmp', img)
    return buf.tobytes()


def capture_loop():
    global raw_frame_buffer, video_writer

    cap = create_pipeline(
        video_source=VIDEO_SOURCE,
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
        fps=TARGET_FPS,
        dev_video_path=DEV_VIDEO_PATH,
    )

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
                time.sleep(0.01)
                continue

            if is_dev_video:
                time.sleep(1.0 / TARGET_FPS)

            with frame_lock:
                raw_frame_buffer = frame.copy()

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
                     fcw_warning, left_sev, right_sev):
    w = BEV_DISPLAY_WIDTH
    h = BEV_DISPLAY_HEIGHT
    im_bev = np.zeros((h, w, 3), dtype=np.uint8)

    s_x = w / (2 * BEV_X_RANGE[1])
    s_y = h / BEV_Y_RANGE[1]

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

        if len(left_pts) >= 2 and len(right_pts) >= 2:
            left_arr = np.array(left_pts, dtype=np.int32)
            right_arr = np.array(right_pts, dtype=np.int32)
            poly = np.vstack((left_arr, right_arr[::-1]))
            overlay = im_bev.copy()
            cv2.fillPoly(overlay, [poly], (120, 60, 30))
            cv2.addWeighted(overlay, 0.4, im_bev, 0.6, 0, dst=im_bev)
            cv2.polylines(im_bev, [left_arr], False, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.polylines(im_bev, [right_arr], False, (220, 220, 220), 2, cv2.LINE_AA)

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


def inference_loop():
    global latest_web_frame, latest_debug_frame, raw_frame_buffer

    print("[INFO] Loading BEVFormer-Tiny model...", flush=True)
    detector = BEVFormerDetector(
        engine_path="models/bevformer_tiny.engine",
        camera_matrix=CAMERA_MATRIX,
        dist_coeff=DIST_COEFF,
    )
    postprocessor = ADASPostprocessor()

    print("[INFO] BEVFormer-Tiny + CuPy ADAS pipeline ready", flush=True)
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

            if state["adas_enabled"]:
                _t_inf = time.perf_counter()
                lane_mask, detections = detector.infer(im0)
                if detections is None:
                    detections = []
                timer.track("inference", time.perf_counter() - _t_inf)

                _t_lane = time.perf_counter()
                left_lane_pts, right_lane_pts = detector.extract_lane_boundaries(lane_mask)

                result = postprocessor.process_lanes(left_lane_pts, right_lane_pts, car_x_meters=0.0)
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

            _t_render = time.perf_counter()
            im_bev = render_bev_frame(
                lane_mask, fcw_overlay_boxes,
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
            _, buf = cv2.imencode('.bmp', im_bev)
            if state["debug_feed_active"]:
                _, buf_debug = cv2.imencode('.bmp', im_debug)
            timer.track("encode", time.perf_counter() - _t_enc)

            with frame_lock:
                latest_web_frame = buf.tobytes()
                if state["debug_feed_active"]:
                    latest_debug_frame = buf_debug.tobytes()

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
    while True:
        with frame_lock:
            frame = latest_web_frame
        if frame is None:
            wait_img = np.zeros((480, 800, 3), dtype=np.uint8)
            cv2.putText(wait_img, "Initializing BEVFormer...",
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            _, buf = cv2.imencode('.bmp', wait_img)
            frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_debug_mjpeg():
    no_feed_frame = None
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
                _, buf = cv2.imencode('.bmp', img)
                no_feed_frame = buf.tobytes()
            frame = no_feed_frame
        elif frame is None:
            wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(wait_img, "Initializing Debug Feed...",
                        (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            _, buf = cv2.imencode('.bmp', wait_img)
            frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)


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
        "twinlite_crop_y": 0,
        "twinlite_crop_h": 416,
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
        "gpu_device_name": "NVIDIA Jetson (BEVFormer-Tiny)",
        "yolop_device": "BEVFormer-Tiny",
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


if __name__ == '__main__':
    print("=" * 50, flush=True)
    print("Edge ADAS v2 — BEVFormer-Tiny + DeepStream + CuPy", flush=True)
    print("=" * 50, flush=True)

    print("Starting Capture thread...", flush=True)
    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_capture.start()

    print("Starting Inference thread...", flush=True)
    t_inference = threading.Thread(target=inference_loop, daemon=True)
    t_inference.start()

    print("Starting Flask server on port 5001...", flush=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
