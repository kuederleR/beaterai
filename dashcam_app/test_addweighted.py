import cv2
import numpy as np

im_bev = np.zeros((400, 640, 3), dtype=np.uint8)
im_bev[100:200, 100:200] = (255, 0, 0)
da_indices = im_bev[:, :, 0] == 255

overlay = np.zeros_like(im_bev)
overlay[da_indices] = (0, 100, 0)
alpha = 0.22

try:
    im_bev[da_indices] = cv2.addWeighted(
        im_bev[da_indices], 1.0 - alpha,
        overlay[da_indices], alpha, 0
    )
    print("Success")
except Exception as e:
    print("Failed:", e)

