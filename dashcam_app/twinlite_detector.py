import cv2
import numpy as np
import os

from trt_runner import TrtRunner, TRT_AVAILABLE


class TwinLiteDetector:
    def __init__(self, engine_path="models/twinlite.engine",
                 crop_y=28, crop_h=360):
        self.input_width = 640
        self.input_height = 360
        self.crop_y = crop_y
        self.crop_h = crop_h
        self.trt_runner = None

        if TRT_AVAILABLE and os.path.exists(engine_path):
            try:
                print(f"[TwinLite] Loading TensorRT engine from {engine_path}", flush=True)
                self.trt_runner = TrtRunner(engine_path)
                print(f"[TwinLite] TensorRT engine ready.", flush=True)
            except Exception as e:
                print(f"[TwinLite] Failed to load TRT engine: {e}", flush=True)
        else:
            print(f"[TwinLite] Engine {engine_path} not found. Auto-building...", flush=True)
            import subprocess
            subprocess.run(["python3", "build_engines.py"])
            if os.path.exists(engine_path):
                self.trt_runner = TrtRunner(engine_path)
                print(f"[TwinLite] TensorRT engine built and ready.", flush=True)

    def _preprocess(self, img):
        h, w = img.shape[:2]
        crop_y = min(self.crop_y, max(0, h - self.crop_h))
        roi = img[crop_y:crop_y + self.crop_h, :]
        roi_resized = cv2.resize(roi, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        normed = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]
        return chw.astype(np.float32)

    def detect(self, img):
        if self.trt_runner is None:
            return None, None

        h, w = img.shape[:2]
        inp = self._preprocess(img)
        outputs = self.trt_runner.infer(inp)
        # outputs[0] = "da" logits, outputs[1] = "ll" logits
        # Each shape: (1, 2, 360, 640) or (2, 360, 640)
        da_raw, ll_raw = outputs[0], outputs[1]

        def _argmax_mask(logits):
            if logits.ndim == 4:
                return np.argmax(logits, axis=1)[0].astype(np.uint8)
            elif logits.ndim == 3:
                return np.argmax(logits, axis=0).astype(np.uint8)
            return np.zeros((self.input_height, self.input_width), dtype=np.uint8)

        da_mask_small = _argmax_mask(da_raw)
        ll_mask_small = _argmax_mask(ll_raw)

        crop_y = min(self.crop_y, max(0, h - self.crop_h))
        da_full = np.zeros((h, w), dtype=np.uint8)
        ll_full = np.zeros((h, w), dtype=np.uint8)

        da_resized = cv2.resize(da_mask_small, (w, self.crop_h),
                                interpolation=cv2.INTER_NEAREST)
        ll_resized = cv2.resize(ll_mask_small, (w, self.crop_h),
                                interpolation=cv2.INTER_NEAREST)

        da_full[crop_y:crop_y + self.crop_h, :] = da_resized
        ll_full[crop_y:crop_y + self.crop_h, :] = ll_resized

        return ll_full, da_full

    @staticmethod
    def lanes_from_mask(ll_mask, car_center_x, min_points=4):
        h, w = ll_mask.shape[:2]
        left_points = []
        right_points = []

        for row in range(h - 1, h // 3, -1):
            row_data = ll_mask[row, :]
            if np.max(row_data) < 128:
                continue
            nonzero = np.where(row_data > 128)[0]
            if len(nonzero) == 0:
                continue
            gaps = np.diff(nonzero) > 4
            segments = np.split(nonzero, np.where(gaps)[0] + 1)
            for seg in segments:
                if len(seg) < 2:
                    continue
                x = float(np.mean(seg))
                y = float(row)
                if x < car_center_x:
                    left_points.append([x, y])
                else:
                    right_points.append([x, y])

        lanes = []
        for pts in (left_points, right_points):
            if len(pts) >= min_points:
                lanes.append(np.array(pts, dtype=np.float32))
        return lanes
