import jetson.inference
import jetson.utils
import cv2
import numpy as np

net = jetson.inference.segNet("fcn-resnet18-cityscapes-512x256")
img = jetson.utils.cudaAllocMapped(width=512, height=256, format="rgba8")
net.Process(img)

mask = jetson.utils.cudaAllocMapped(width=512, height=256, format="gray8")
net.Mask(mask, 512, 256)
jetson.utils.cudaDeviceSynchronize()

mask_np = jetson.utils.cudaToNumpy(mask)
unique, counts = np.unique(mask_np, return_counts=True)
print("Unique classes in gray8 mask:", dict(zip(unique, counts)))
