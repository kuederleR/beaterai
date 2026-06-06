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

# With u_z[1] > 0
u_z_up = np.cross(u_x, u_y)
if u_z_up[1] > 0:
    u_z_up = -u_z_up

# With u_z[1] < 0 (pointing down)
u_z_down = np.cross(u_x, u_y)
if u_z_down[1] < 0:
    u_z_down = -u_z_down

h = 1.2192

print("USING u_z_down (pointing downwards):")
M_down = np.stack([u_x, u_y, h * u_z_down], axis=1)
H_down = K_INFER @ M_down

for Y in [2.0, 10.0, 100.0, 1000.0]:
    road_point = np.array([0, Y, 1.0])
    img_h = H_down @ road_point
    img_pt = img_h / img_h[2]
    print(f"{Y}m ahead image pt: {img_pt}")

print("\nUSING u_z_up (current code):")
M_up = np.stack([u_x, u_y, h * u_z_up], axis=1)
H_up = K_INFER @ M_up

for Y in [2.0, 10.0, 100.0, 1000.0]:
    road_point = np.array([0, Y, 1.0])
    img_h = H_up @ road_point
    img_pt = img_h / img_h[2]
    print(f"{Y}m ahead image pt: {img_pt}")

