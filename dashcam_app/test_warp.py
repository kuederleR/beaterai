import numpy as np

K_INFER = np.array([
    [449.3, 0.0, 336.7],
    [0.0, 449.3, 174.6],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

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

h = 1.2192
M = np.stack([u_x, u_y, h * u_z], axis=1)
H = K_INFER @ M
H_inv = np.linalg.inv(H)

BEV_WIDTH = 640
BEV_HEIGHT = 400
X_MIN = -6.0
X_MAX = 6.0
Y_MIN = 1.0
Y_MAX = 100.0

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

print("H_cam2bev:\n", H_cam2bev)

# map a point near bottom of image (e.g. u=336, v=390)
img_pt = np.array([336.7, 390.0, 1.0])
bev_pt = H_cam2bev @ img_pt
bev_pt = bev_pt / bev_pt[2]
print("Bottom of image maps to BEV:", bev_pt)

