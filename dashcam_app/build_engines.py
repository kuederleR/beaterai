"""
Build TensorRT engines for YOLOPv2 (FP16 / INT8).

Three stages:
  1. export ONNX from PyTorch JIT model
  2. build INT8 calibration cache from a video
  3. build engine with trtexec (FP16 or INT8)

All stages are idempotent — skip if output already exists.
"""

import os
import sys
import time
import subprocess
import numpy as np
import cv2

MODEL_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
IMG_SIZE = 480


# ---------------------------------------------------------------------------
# Stage 1 — ONNX export
# ---------------------------------------------------------------------------

def _flatten_onnx_sequences(onnx_path):
    import onnx
    from onnx import helper, TensorProto
    model = onnx.load(onnx_path)
    graph = model.graph
    seq_outputs = {}
    graph_output_names = set(o.name for o in graph.output)
    for node in list(graph.node):
        if node.op_type == 'SequenceConstruct':
            if node.output[0] in graph_output_names:
                seq_outputs[node.output[0]] = list(node.input)
                graph.node.remove(node)
    original_outputs = list(graph.output)
    del graph.output[:]
    for output in original_outputs:
        if output.name in seq_outputs:
            for inp_name in seq_outputs[output.name]:
                val_info = next(
                    (vi for vi in graph.value_info if vi.name == inp_name), None
                )
                shape = None
                if val_info and val_info.type.tensor_type.shape:
                    shape = [d.dim_value for d in val_info.type.tensor_type.shape.dim]
                new_out = helper.make_tensor_value_info(
                    inp_name, TensorProto.FLOAT, shape
                )
                graph.output.append(new_out)
        else:
            graph.output.append(output)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)


def export_yolop_onnx(model_path="data/weights/yolopv2.pt",
                      onnx_path="data/weights/yolopv2.onnx"):
    if os.path.exists(onnx_path):
        print(f"[build] ONNX exists at {onnx_path}, skipping export", flush=True)
        return

    import torch
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[build] Downloading YOLOPv2 weights from {MODEL_URL} ...", flush=True)
        import urllib.request
        req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(model_path, 'wb') as f:
            f.write(resp.read())
        print("[build] Download complete", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build] Loading model on {device} ...", flush=True)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model = model.float()
    dtype = torch.float32

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=dtype, device=device)
    print(f"[build] Exporting ONNX to {onnx_path} ...", flush=True)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["pred", "ag0", "ag1", "ag2", "seg", "ll"],
        opset_version=12,
        do_constant_folding=True,
    )
    print("[build] Flattening SequenceConstruct nodes for TensorRT ...", flush=True)
    _flatten_onnx_sequences(onnx_path)
    print("[build] ONNX export done.", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 — INT8 calibration cache
# ---------------------------------------------------------------------------

class _VideoCalibrator:
    def __init__(self, video_path, batch_size=1, img_size=480, cache_path=None):
        import tensorrt as trt
        self._trt = trt
        self.batch_size = batch_size
        self.img_size = img_size
        self.cache_path = cache_path

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            self._total = 500
        self._idx = 0

        self._device_input = None

    def __del__(self):
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self._idx >= self._total:
            return None

        batch = []
        for _ in range(self.batch_size):
            if self._idx >= self._total:
                break
            ret, frame = self._cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normed = rgb.astype(np.float32) / 255.0
            chw = np.transpose(normed, (2, 0, 1))
            batch.append(chw)
            self._idx += 1

        if not batch:
            return None

        arr = np.stack(batch, axis=0).astype(np.float32)
        if self._device_input is None or self._device_input.shape != arr.shape:
            import torch
            self._device_input = torch.empty(arr.shape, dtype=torch.float32,
                                              device='cuda')
        self._device_input.copy_(torch.from_numpy(arr).cuda())
        return [int(self._device_input.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                f.write(cache)
            print(f"[build] Calibration cache saved to {self.cache_path}", flush=True)


def build_int8_calibration_cache(onnx_path="data/weights/yolopv2.onnx",
                                 video_path=None,
                                 cache_path="data/weights/yolopv2_int8.cache",
                                 max_calib_frames=200):
    if os.path.exists(cache_path):
        print(f"[build] Calibration cache exists at {cache_path}, skipping", flush=True)
        return

    if not video_path or not os.path.exists(video_path):
        print("[build] No calibration video available, building INT8 engine without cache",
              flush=True)
        return

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] Parsing {onnx_path} for calibration ...", flush=True)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"[build] ONNX parse error: {parser.get_error(err)}", flush=True)
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = _VideoCalibrator(video_path, batch_size=1, img_size=IMG_SIZE,
                                  cache_path=cache_path)
    config.int8_calibrator = calibrator

    print(f"[build] Building calibration engine (up to {max_calib_frames} frames) ...",
          flush=True)
    builder.build_serialized_network(network, config)
    print("[build] Calibration done.", flush=True)


# ---------------------------------------------------------------------------
# Stage 3 — INT8 engine with trtexec
# ---------------------------------------------------------------------------

def build_yolop_int8_engine(onnx_path="data/weights/yolopv2.onnx",
                            cache_path="data/weights/yolopv2_int8.cache",
                            engine_path="data/weights/yolopv2_int8.engine"):
    if os.path.exists(engine_path):
        print(f"[build] INT8 engine exists at {engine_path}, skipping", flush=True)
        return

    for needed in (onnx_path,):
        if not os.path.exists(needed):
            raise RuntimeError(f"Required file not found: {needed}")

    cmd = ["trtexec",
           f"--onnx={onnx_path}",
           f"--saveEngine={engine_path}",
           "--int8",
           "--memPoolSize=workspace:2048"]

    if cache_path and os.path.exists(cache_path):
        cmd.append(f"--calib={cache_path}")
        print(f"[build] Using calibration cache: {cache_path}", flush=True)
    else:
        print("[build] No calibration cache — trtexec will use random calibration",
              flush=True)

    print(f"[build] Building INT8 engine via trtexec ...", flush=True)
    print(f"[build] Command: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(f"[trtexec] {line.rstrip()}", flush=True)
    proc.wait(timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed with code {proc.returncode}")
    print(f"[build] INT8 engine saved to {engine_path}", flush=True)


# ---------------------------------------------------------------------------
# Convenience: ensure all stages
# ---------------------------------------------------------------------------

def ensure_yolop_int8_engine(model_path="data/weights/yolopv2.pt",
                             onnx_path="data/weights/yolopv2.onnx",
                             cache_path="data/weights/yolopv2_int8.cache",
                             engine_path="data/weights/yolopv2_int8.engine",
                             calib_video_path=None):
    if os.path.exists(engine_path):
        print(f"[build] INT8 engine ready at {engine_path}", flush=True)
        return engine_path

    print("=" * 60, flush=True)
    print("[build] YOLOPv2 INT8 engine not found — building now.", flush=True)
    print("[build] This may take 5–15 minutes on Jetson Orin Nano.", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    export_yolop_onnx(model_path, onnx_path)

    if not os.path.exists(cache_path):
        try:
            build_int8_calibration_cache(onnx_path, calib_video_path, cache_path)
        except Exception as e:
            print(f"[build] Calibration skipped: {e}", flush=True)

    build_yolop_int8_engine(onnx_path, cache_path, engine_path)
    elapsed = time.time() - t0
    print(f"[build] INT8 engine build complete in {elapsed:.0f}s.", flush=True)
    return engine_path


if __name__ == "__main__":
    ensure_yolop_int8_engine(calib_video_path=os.environ.get("DEV_VIDEO_PATH"))
