import numpy as np

K_INFER = np.array([
    [449.3, 0.0, 336.7],
    [0.0, 449.3, 174.6],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

vp_x = K_INFER[0, 2]
vp_y = 160.0 # From the screenshot, vp is probably around 50, but let's test 160 first.
    
K_inv = np.linalg.inv(K_INFER)
u_y = K_inv @ np.array([vp_x, vp_y, 1.0], dtype=np.float32)
u_y = u_y / np.linalg.norm(u_y)

u_x = np.array([u_y[2], 0.0, -u_y[0]], dtype=np.float32)
u_x = u_x / np.linalg.norm(u_x)

u_z = np.cross(u_x, u_y)
if u_z[1] < 0:
    u_z = -u_z

CAMERA_HEIGHT = 1.2192
M = np.stack([u_x, u_y, CAMERA_HEIGHT * u_z], axis=1)
H = K_INFER @ M
H_inv = np.linalg.inv(H)

def image_to_road(pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    road_h = (H_inv @ pts_h.T).T
    valid = road_h[:, 2] > 1e-5
    road = np.zeros((len(pts), 2), dtype=np.float32)
    road[valid, 0] = road_h[valid, 0] / road_h[valid, 2]
    road[valid, 1] = road_h[valid, 1] / road_h[valid, 2]
    road[~valid] = np.nan
    return road

# Create some dummy left lane line points in the image
# Let's say it goes from bottom left to vanishing point
vs = np.linspace(400, 200, 20)
us = np.linspace(100, 300, 20)
pts = np.stack([us, vs], axis=1).astype(np.float32)

pts_road = image_to_road(pts)
print("Image to road pts:\n", pts_road[:5])

# Now let's try with vp_y = 50 (like in the screenshot)
print("\n--- Testing with vp_y = 50 ---")
vp_y = 50.0
u_y = K_inv @ np.array([vp_x, vp_y, 1.0], dtype=np.float32)
u_y = u_y / np.linalg.norm(u_y)
u_x = np.array([u_y[2], 0.0, -u_y[0]], dtype=np.float32)
u_x = u_x / np.linalg.norm(u_x)
u_z = np.cross(u_x, u_y)
if u_z[1] < 0:
    u_z = -u_z
M = np.stack([u_x, u_y, CAMERA_HEIGHT * u_z], axis=1)
H = K_INFER @ M
H_inv = np.linalg.inv(H)

pts_road_50 = image_to_road(pts)
print("Image to road pts (vp_y=50):\n", pts_road_50[:5])

