import numpy as np

K_INFER = np.array([
    [449.3, 0.0, 336.7],
    [0.0, 449.3, 174.6],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

vp_x = K_INFER[0, 2] # Default center
vp_y = 160.0 # Default vanishing point y (approx 1.86 deg pitch down)
    
K_inv = np.linalg.inv(K_INFER)

# Forward unit vector in camera coordinates
u_y = K_inv @ np.array([vp_x, vp_y, 1.0], dtype=np.float32)
u_y = u_y / np.linalg.norm(u_y)

# Right unit vector (horizontal in camera, roll=0)
u_x = np.array([u_y[2], 0.0, -u_y[0]], dtype=np.float32)
u_x = u_x / np.linalg.norm(u_x)

# Up/Normal unit vector
u_z = np.cross(u_x, u_y)
if u_z[1] > 0:
    u_z = -u_z
    
CAMERA_HEIGHT = 1.2192
h = CAMERA_HEIGHT

print("u_x:", u_x)
print("u_y:", u_y)
print("u_z:", u_z)
print("Road origin (camera coords):", h * u_z)

M = np.stack([u_x, u_y, h * u_z], axis=1)
H = K_INFER @ M
H_inv = np.linalg.inv(H)

print("H:\n", H)

# Let's project a road point to the image
road_point = np.array([0, 10.0, 1.0]) # 10 meters ahead
img_h = H @ road_point
img_pt = img_h / img_h[2]
print("10m ahead image pt:", img_pt)

