import os
import json
import threading
import subprocess
import cv2
import numpy as np

from trt_runner import TrtRunner, TRT_AVAILABLE

BEV_GRID_SIZE = 100
BEV_X_RANGE = (-30.0, 30.0)
BEV_Y_RANGE = (-15.0, 15.0)
DEFAULT_ENGINE_PATH = "models/bevformer_tiny.engine"
EXTRINSICS_CONFIG_PATH = "models/bevformer_extrinsics.json"


def _load_camera_intrinsics():
    for path in ("calibration.yaml", os.path.join(os.path.dirname(__file__), "calibration.yaml")):
        if not os.path.exists(path):
            continue
        try:
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                K = fs.getNode("camera_matrix").mat()
                D = fs.getNode("dist_coeff").mat()
                fs.release()
                if K is not None and D is not None:
                    return K.astype(np.float32), D.astype(np.float32)
        except Exception:
            pass

    print("[BEVFormer] Using default nuScenes camera intrinsics", flush=True)
    K = np.array([
        [1266.417, 0.0, 816.268],
        [0.0, 1266.417, 491.507],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    D = np.zeros((1, 5), dtype=np.float32)
    return K, D


def _load_extrinsics_override():
    if not os.path.exists(EXTRINSICS_CONFIG_PATH):
        return None
    try:
        with open(EXTRINSICS_CONFIG_PATH, "r") as f:
            data = json.load(f)
        if "extrinsics" in data:
            M = np.array(data["extrinsics"], dtype=np.float32)
            print(f"[BEVFormer] Loaded extrinsics override from {EXTRINSICS_CONFIG_PATH}", flush=True)
            return M
    except Exception as e:
        print(f"[BEVFormer] Failed to load extrinsics override: {e}", flush=True)
    return None


class BEVFormerDetector:
    def __init__(self, engine_path=DEFAULT_ENGINE_PATH,
                 camera_matrix=None, dist_coeff=None,
                 extrinsics_override=None):
        self.trt_runner = None
        self.bev_grid = None

        K, D = _load_camera_intrinsics()
        self.camera_matrix = camera_matrix if camera_matrix is not None else K
        self.dist_coeff = dist_coeff if dist_coeff is not None else D

        self.extrinsics = extrinsics_override if extrinsics_override is not None else _load_extrinsics_override()
        if self.extrinsics is None:
            self.extrinsics = self._nuscenes_default_extrinsics()
            print("[BEVFormer] Using default nuScenes front-camera extrinsics", flush=True)

        self.input_width = 1600
        self.input_height = 900
        self.bev_grid_size = BEV_GRID_SIZE

        self._precompute_bev_grid()

        if TRT_AVAILABLE and os.path.exists(engine_path):
            try:
                self.trt_runner = TrtRunner(engine_path)
                print(f"[BEVFormer] TensorRT engine loaded from {engine_path}", flush=True)
            except Exception as e:
                print(f"[BEVFormer] Failed to load TRT engine: {e}", flush=True)
                self.trt_runner = None
        else:
            print(f"[BEVFormer] Engine {engine_path} not found. Starting auto-build...", flush=True)
            self._auto_build(engine_path)

    def _auto_build(self, engine_path):
        def _build():
            build_dir = "models/bevformer_build"
            os.makedirs(build_dir, exist_ok=True)
            onnx_path = os.path.join(build_dir, "bevformer_tiny_fixed.onnx")
            try:
                if not os.path.exists(onnx_path):
                    print("[BEVFormer] Downloading pre-exported ONNX from HuggingFace...", flush=True)
                    import urllib.request
                    url = ("https://huggingface.co/AXERA-TECH/bevformer/resolve/main/"
                           "bevformer_tiny_fixed.onnx")
                    urllib.request.urlretrieve(url, onnx_path)

                print("[BEVFormer] Building TensorRT engine (FP16)...", flush=True)
                subprocess.run([
                    "trtexec",
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    "--fp16",
                    "--memPoolSize=workspace:2048",
                    "--inputIChannels=3",
                    "--inputHW=900,1600",
                ], check=True, timeout=900)
                print(f"[BEVFormer] Engine saved to {engine_path}. Reloading...", flush=True)
                self.trt_runner = TrtRunner(engine_path)
            except Exception as e:
                print(f"[BEVFormer] Auto-build failed: {e}", flush=True)
                print("[BEVFormer] Build manually: download ONNX from "
                      "https://huggingface.co/AXERA-TECH/bevformer/blob/main/bevformer_tiny_fixed.onnx"
                      ", then run: trtexec --onnx=bevformer_tiny_fixed.onnx "
                      "--saveEngine=models/bevformer_tiny.engine --fp16", flush=True)
        t = threading.Thread(target=_build, daemon=True)
        t.start()

    @staticmethod
    def _nuscenes_default_extrinsics():
        M = np.eye(4, dtype=np.float32)
        M[0, 3] = 1.5
        M[1, 3] = 0.0
        M[2, 3] = 1.6
        return M

    def _precompute_bev_grid(self):
        xs = np.linspace(BEV_X_RANGE[0], BEV_X_RANGE[1], self.bev_grid_size)
        ys = np.linspace(BEV_Y_RANGE[0], BEV_Y_RANGE[1], self.bev_grid_size)
        self.bev_grid = np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32)

    def _preprocess(self, img_bgr):
        if img_bgr.shape[1] != self.input_width or img_bgr.shape[0] != self.input_height:
            img = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            h, w = img_bgr.shape[:2]
            scale = min(self.input_width / w, self.input_height / h)
            nw = int(w * scale)
            nh = int(h * scale)
            resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            dx = (self.input_width - nw) // 2
            dy = (self.input_height - nh) // 2
            img[dy:dy+nh, dx:dx+nw] = resized
        else:
            img = img_bgr.copy()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std
        chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
        return chw.astype(np.float32)

    def infer(self, img_bgr):
        if self.trt_runner is None:
            return None, None

        inp = self._preprocess(img_bgr)
        outputs = self.trt_runner.infer(inp)

        lane_mask = None
        detections = []

        for arr in outputs:
            s = arr.shape
            if len(s) == 4 and s[1] == 1 and s[2] == self.bev_grid_size and s[3] == self.bev_grid_size:
                probs = 1.0 / (1.0 + np.exp(-arr))
                lane_mask = (probs[0, 0] > 0.5).astype(np.uint8)
            elif len(s) == 3 and s[1] >= 7:
                detection_arr = arr[0]
                for i in range(detection_arr.shape[0]):
                    if detection_arr[i, 0] > 0.3:
                        cx, cy, cz, w, l, h, heading = detection_arr[i, 1:8]
                        detections.append({
                            "center_x": float(cx),
                            "center_y": float(cy),
                            "center_z": float(cz),
                            "width": float(w),
                            "length": float(l),
                            "height": float(h),
                            "heading": float(heading),
                            "score": float(detection_arr[i, 0]),
                        })

        return lane_mask, detections

    def bev_to_road(self, bev_u, bev_v):
        if self.bev_grid is None:
            return None
        u_idx = int(np.clip(bev_u, 0, self.bev_grid_size - 1))
        v_idx = int(np.clip(bev_v, 0, self.bev_grid_size - 1))
        return self.bev_grid[v_idx, u_idx]

    def extract_lane_boundaries(self, lane_mask):
        if lane_mask is None:
            return None, None

        h, w = lane_mask.shape[:2]
        left_points = []
        right_points = []

        for v in range(h):
            row = lane_mask[v, :]
            nonzero = np.where(row > 0)[0]
            if len(nonzero) == 0:
                continue
            left_u = nonzero[0]
            right_u = nonzero[-1]

            left_road = self.bev_to_road(left_u, v)
            right_road = self.bev_to_road(right_u, v)

            if left_road is not None:
                left_points.append(left_road)
            if right_road is not None:
                right_points.append(right_road)

        left_arr = np.array(left_points, dtype=np.float32) if left_points else None
        right_arr = np.array(right_points, dtype=np.float32) if right_points else None
        return left_arr, right_arr
