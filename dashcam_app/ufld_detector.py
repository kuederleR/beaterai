import cv2
import numpy as np
import os

from trt_runner import TrtRunner, TRT_AVAILABLE


class ULFDLaneDetector:
    def __init__(self, engine_path="models/ufld.engine"):
        self.input_height = 288
        self.input_width = 800
        self.num_lanes = 4
        self.num_row = 56
        self.num_col = 41
        self.exist_threshold = 0.3

        row_anchor_frac = np.linspace(0.42, 1.0, self.num_row)
        self.row_anchor = (row_anchor_frac * self.input_height).astype(np.float32)
        self.col_sample = np.linspace(0, self.input_width - 1, self.num_col).astype(np.float32)

        self.trt_runner = None
        self.model = None

        if TRT_AVAILABLE and os.path.exists(engine_path):
            try:
                print(f"[UFLD] Loading TensorRT engine from {engine_path}", flush=True)
                self.trt_runner = TrtRunner(engine_path)
                print(f"[UFLD] TensorRT engine ready.", flush=True)
            except Exception as e:
                print(f"[UFLD] Failed to load TRT engine: {e}", flush=True)
        elif not TRT_AVAILABLE:
            print(f"[UFLD] TensorRT not available. Loading PyTorch fallback.", flush=True)
            self._load_pytorch_fallback()
        else:
            print(f"[UFLD] Engine file {engine_path} not found. Auto-building engine...", flush=True)
            import subprocess
            subprocess.run(["python3", "build_engines.py"])
            if os.path.exists(engine_path):
                self.trt_runner = TrtRunner(engine_path)
                print(f"[UFLD] TensorRT engine built and ready.", flush=True)
            else:
                self._load_pytorch_fallback()
    def _load_pytorch_fallback(self):
        print("[UFLD] No TensorRT engine found in models/ufld.engine.", flush=True)
        print("[UFLD] Run python3 build_engines.py to build it, or place a pre-built engine.", flush=True)
        self.model = None

    def _preprocess(self, img):
        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std
        img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
        return img_chw

    def _decode_output(self, output, img_width, img_height):
        if output.ndim == 4 and output.shape[1] == self.num_lanes:
            out = output[0]
        elif output.ndim == 2:
            out = output[0].reshape(self.num_lanes, self.num_row, self.num_col + 1)
        elif output.ndim == 3:
            out = output
        else:
            print(f"[UFLD] Unexpected output shape: {output.shape}", flush=True)
            return []

        scale_x = img_width / self.input_width
        scale_y = img_height / self.input_height

        lanes = []
        for lane_idx in range(self.num_lanes):
            pts = []
            for row_idx in range(self.num_row):
                cls_logits = out[lane_idx, row_idx, :self.num_col]
                exist_logit = out[lane_idx, row_idx, self.num_col]
                if exist_logit < self.exist_threshold:
                    continue
                cls_prob = np.exp(cls_logits - np.max(cls_logits))
                cls_prob = cls_prob / cls_prob.sum()
                col_idx = np.argmax(cls_prob)
                x_uf = self.col_sample[col_idx] * scale_x
                y_uf = self.row_anchor[row_idx] * scale_y
                pts.append((float(x_uf), float(y_uf)))
            if len(pts) >= 4:
                lanes.append(np.array(pts, dtype=np.float32))
        return lanes

    def detect(self, img):
        if self.model is None and self.trt_runner is None:
            return []

        h_orig, w_orig = img.shape[:2]
        img_chw = self._preprocess(img)

        if self.trt_runner is not None:
            out = self.trt_runner.infer(img_chw)
            raw = out[0]
        else:
            import torch
            img_tensor = torch.from_numpy(img_chw).cuda()
            with torch.inference_mode():
                raw = self.model(img_tensor).cpu().numpy()

        lanes = self._decode_output(raw, w_orig, h_orig)
        return lanes
