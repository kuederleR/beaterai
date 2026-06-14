import numpy as np

CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    pass


def _to_gpu(arr):
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr


def _to_cpu(arr):
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def fit_lane_polygon(lane_points, degree=3, y_min=0.0, y_max=50.0):
    if lane_points is None or len(lane_points) < degree + 1:
        return None, None, None

    pts = np.asarray(lane_points, dtype=np.float32)
    X = pts[:, 0]
    Y = pts[:, 1]

    order = np.argsort(Y)
    X = X[order]
    Y = Y[order]

    x_gpu = _to_gpu(X)
    y_gpu = _to_gpu(Y)

    if CUPY_AVAILABLE:
        coeffs = cp.polyfit(y_gpu, x_gpu, degree)
    else:
        coeffs = np.polyfit(Y, X, degree)

    eval_y = np.arange(y_min, y_max, 1.0, dtype=np.float32)
    eval_y_gpu = _to_gpu(eval_y)
    eval_x = np.polyval(_to_cpu(coeffs) if CUPY_AVAILABLE else coeffs, eval_y)

    tangent_x = 0.0
    heading_error = 0.0
    if degree >= 1:
        deriv = np.polyder(coeffs)
        tangent_x = np.polyval(deriv, 0.0)
        heading_error = float(np.arctan(tangent_x))

    return _to_cpu(coeffs) if CUPY_AVAILABLE else coeffs, eval_x, heading_error


def compute_ego_offset(left_coeffs, right_coeffs, eval_y=0.0):
    if left_coeffs is None or right_coeffs is None:
        return None, None

    left_x = np.polyval(left_coeffs, eval_y)
    right_x = np.polyval(right_coeffs, eval_y)
    lane_center = 0.5 * (left_x + right_x)
    lane_width = right_x - left_x
    return float(lane_center), float(lane_width)


def compute_lane_position(left_coeffs, right_coeffs, car_x=0.0, eval_y=0.0):
    if left_coeffs is None or right_coeffs is None:
        return 0.5, 0.0

    left_x = np.polyval(left_coeffs, eval_y)
    right_x = np.polyval(right_coeffs, eval_y)
    lane_width = right_x - left_x

    if lane_width < 0.5:
        return 0.5, 0.0

    pos = (car_x - left_x) / lane_width
    pos = float(np.clip(pos, 0.0, 1.0))
    return pos, float(lane_width)


def compute_severity(distance, comfort_dist=1.1, threshold=0.3):
    if distance is None or distance > comfort_dist:
        return 0.0
    severity = (comfort_dist - distance) / threshold
    return float(np.clip(severity, 0.0, 1.0))


def smooth_poly(new_coeffs, history, max_history=8):
    if new_coeffs is not None:
        history.append(new_coeffs)
        if len(history) > max_history:
            history.pop(0)
    if len(history) > 0:
        return np.mean(history, axis=0)
    return None


class ADASPostprocessor:
    def __init__(self):
        self.left_poly_history = []
        self.right_poly_history = []
        self.car_offset_ema = 0.0

    def reset(self):
        self.left_poly_history.clear()
        self.right_poly_history.clear()
        self.car_offset_ema = 0.0

    def process_lanes(self, left_lane_points, right_lane_points, car_x_meters=0.0):
        left_coeffs_raw, left_eval, left_heading = fit_lane_polygon(left_lane_points)
        right_coeffs_raw, right_eval, right_heading = fit_lane_polygon(right_lane_points)

        left_coeffs = smooth_poly(left_coeffs_raw, self.left_poly_history)
        right_coeffs = smooth_poly(right_coeffs_raw, self.right_poly_history)

        ego_center, lane_width = compute_ego_offset(left_coeffs, right_coeffs, eval_y=5.0)

        if ego_center is not None and lane_width is not None and lane_width > 1.0:
            offset = ego_center - car_x_meters
            self.car_offset_ema = 0.95 * self.car_offset_ema + 0.05 * offset
            compensated_x = car_x_meters + self.car_offset_ema
        else:
            compensated_x = car_x_meters

        lane_position, lw = compute_lane_position(
            left_coeffs, right_coeffs, compensated_x, eval_y=5.0
        )

        return {
            "left_coeffs": left_coeffs,
            "right_coeffs": right_coeffs,
            "left_eval_x": left_eval,
            "right_eval_x": right_eval,
            "lane_position": lane_position,
            "lane_width": lw if lw > 0 else lane_width or 0.0,
            "ego_center": ego_center,
            "compensated_x": compensated_x,
            "heading_error": left_heading if left_heading is not None else 0.0,
        }

    def assess_fcw(self, detections, compensated_x, warning_distance=15.0, lane_half_width=1.6):
        threats = []
        triggered = False
        for det in detections:
            cx = det["center_x"]
            cy = det["center_y"]
            if abs(cx - compensated_x) > lane_half_width:
                continue
            if cy > warning_distance:
                continue
            triggered = True
            threats.append(det)
        return triggered, threats

    def assess_ldw(self, left_coeffs, right_coeffs, compensated_x,
                   left_comfort=1.1, right_comfort=1.1):
        left_sev = 0.0
        right_sev = 0.0

        y_eval = 5.0
        if left_coeffs is not None:
            left_x = np.polyval(left_coeffs, y_eval)
            d_left = compensated_x - left_x
        else:
            d_left = 999.0

        if right_coeffs is not None:
            right_x = np.polyval(right_coeffs, y_eval)
            d_right = right_x - compensated_x
        else:
            d_right = 999.0

        left_sev = compute_severity(d_left, left_comfort)
        right_sev = compute_severity(d_right, right_comfort)

        triggered = left_sev > 0.8 or right_sev > 0.8
        return triggered, left_sev, right_sev
