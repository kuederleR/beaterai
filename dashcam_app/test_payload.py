import numpy as np

BEV_WIDTH = 640
BEV_HEIGHT = 400
X_MIN = -6.0
X_MAX = 6.0
Y_MIN = 1.0
Y_MAX = 100.0

def road_to_bev(pts):
    s_x = (BEV_WIDTH - 1) / (X_MAX - X_MIN)
    s_y = (BEV_HEIGHT - 1) / (Y_MAX - Y_MIN)
    u_bev = (pts[:, 0] - X_MIN) * s_x
    v_bev = (Y_MAX - pts[:, 1]) * s_y
    return np.stack([u_bev, v_bev], axis=1)

def build_lane_overlay_payload(left_poly, right_poly):
    eval_ys = np.arange(1.0, 100.0, 1.0, dtype=np.float32)
    
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

    # The issue: In dashcam.py, left_points and right_points are overwritten here!
    left_points = left_pts_bev.tolist()
    right_points = right_pts_bev.tolist()
    center_points = center_pts_bev.tolist()

    polygon = []
    if len(left_points) >= 2 and len(right_points) >= 2:
        polygon = left_points + list(reversed(right_points))

    return polygon

# Fake a perfect straight lane
left_poly = np.polyfit(np.arange(1, 100), np.full(99, -2.0), 2)
right_poly = np.polyfit(np.arange(1, 100), np.full(99, 2.0), 2)

polygon = build_lane_overlay_payload(left_poly, right_poly)
print("First 5 polygon points:", polygon[:5])
print("Last 5 polygon points:", polygon[-5:])
