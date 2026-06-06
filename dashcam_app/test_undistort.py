import cv2
import numpy as np

K_INFER = np.array([
    [449.3456840466663, 0.0, 336.72375692634625],
    [0.0, 449.36500344504043, 174.6203780612756],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFF = np.array([0.027602996212313838, -0.064486646048556584,
                       0.0034829585578156821, -0.0048244561182151577,
                       0.035676429431834245], dtype=np.float32)

# Create a dummy mask with a straight diagonal line
ll_mask = np.zeros((400, 640), dtype=np.uint8)
for i in range(400):
    j = int(i * 1.5)
    if j < 640:
        ll_mask[i, j] = 1

ll_mask_255 = ll_mask * 255
ll_undist = cv2.undistort(ll_mask_255, K_INFER, DIST_COEFF)
ll_undist_bin = (ll_undist > 127).astype(np.uint8)

# Check if it's contiguous
import sys
for i in range(100, 300, 20):
    row = ll_undist_bin[i, :]
    cols = np.where(row > 0)[0]
    print(f"Row {i}: {cols}")

