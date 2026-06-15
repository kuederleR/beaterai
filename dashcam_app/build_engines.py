"""
Build TensorRT engines for YOLOPv2.

Two stages:
  1. export ONNX from PyTorch JIT model
  2. build engine with trtexec (FP16 or INT8)

All stages are idempotent — skip if output already exists.
"""

import os
import sys
import time
import subprocess

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
# Stage 2 — TensorRT engine with trtexec
# ---------------------------------------------------------------------------

TRT_PRECISION_FLAGS = {
    "fp16": "--fp16",
    "int8": "--int8",
}


def build_yolop_trt_engine(onnx_path="data/weights/yolopv2.onnx",
                           engine_path="data/weights/yolopv2_fp16.engine",
                           precision="fp16"):
    if os.path.exists(engine_path):
        print(f"[build] {precision.upper()} engine exists at {engine_path}, skipping",
              flush=True)
        return

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX not found: {onnx_path}")

    precision_flag = TRT_PRECISION_FLAGS.get(precision)
    if precision_flag is None:
        raise ValueError(f"Unsupported precision: {precision!r}")

    cmd = ["trtexec",
           f"--onnx={onnx_path}",
           f"--saveEngine={engine_path}",
           precision_flag,
           "--memPoolSize=workspace:2048"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        pass
    proc.wait(timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed with code {proc.returncode}")
    print(f"[build] {precision.upper()} engine saved to {engine_path}", flush=True)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def ensure_yolop_trt_engine(model_path="data/weights/yolopv2.pt",
                            onnx_path="data/weights/yolopv2.onnx",
                            engine_path="data/weights/yolopv2_fp16.engine",
                            precision="fp16"):
    if os.path.exists(engine_path):
        return engine_path

    print("=" * 60, flush=True)
    print(f"[build] YOLOPv2 {precision.upper()} engine not found — building now.",
          flush=True)
    print(f"[build] This may take 2–3 minutes on Jetson Orin Nano.", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    export_yolop_onnx(model_path, onnx_path)
    build_yolop_trt_engine(onnx_path, engine_path, precision)
    elapsed = time.time() - t0
    return engine_path


if __name__ == "__main__":
    import os
    precision = os.environ.get("TRT_PRECISION", "fp16")
    engine_path = f"data/weights/yolopv2_{precision}.engine"
    ensure_yolop_trt_engine(engine_path=engine_path, precision=precision)
