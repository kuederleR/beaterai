"""
Build TensorRT engines for FCN-ResNet18 and UFLD models.

Can run inside the Docker container or on the host Jetson.

Inside container (engines persist via ./models volume):
    docker compose run --rm dashcam python3 build_engines.py

Or on the host Jetson directly:
    python3 build_engines.py

Requirements inside container (already satisfied):
    - torch, torchvision  (nvcr.io base image)
    - tensorrt            (nvcr.io base image)
    - internet for torchhub UFLD model download

Output:
    models/fcn_resnet18.engine
    models/ufld.engine
"""

import os
import sys

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
# TensorRT Python API builder (no trtexec needed)
# ---------------------------------------------------------------------------

def build_engine_from_onnx(onnx_path, engine_path, fp16=True):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] Parsing {onnx_path} ...", flush=True)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"[build] ONNX parse error: {parser.get_error(err)}", flush=True)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled", flush=True)
    else:
        print("[build] FP16 not available on this platform, using FP32", flush=True)

    print(f"[build] Building engine (this takes a few minutes)...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[build] Engine build failed", flush=True)
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build] Engine saved to {engine_path}", flush=True)


# ---------------------------------------------------------------------------
# FCN-ResNet18
# ---------------------------------------------------------------------------

def build_fcn_engine():
    engine_path = "models/fcn_resnet18.engine"
    onnx_path = "models/fcn_resnet18.onnx"

    if os.path.exists(engine_path):
        print(f"[build] {engine_path} exists, skipping")
        return

    print("[build] Exporting FCN-ResNet18 to ONNX ...", flush=True)
    import torch
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for attempt in range(3):
        try:
            if attempt == 0:
                w = torchvision.models.segmentation.FCN_ResNet18_Weights.CITYSCAPES_512x256
                model = torchvision.models.segmentation.fcn_resnet18(weights=w)
            elif attempt == 1:
                model = torchvision.models.segmentation.fcn_resnet18(weights="CITYSCAPES_512x256")
            else:
                model = torchvision.models.segmentation.fcn_resnet18(pretrained=True)
            break
        except (AttributeError, TypeError, ValueError):
            continue
    else:
        print("[build] Could not load FCN-ResNet18 model", flush=True)
        sys.exit(1)

    model = model.eval().to(device)

    dummy = torch.zeros(1, 3, 256, 512, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"[build] ONNX saved to {onnx_path}", flush=True)
    build_engine_from_onnx(onnx_path, engine_path)


# ---------------------------------------------------------------------------
# UFLD  (RESA)
# ---------------------------------------------------------------------------

def build_ufld_engine():
    engine_path = "models/ufld.engine"
    onnx_path = "models/ufld.onnx"

    if os.path.exists(engine_path):
        print(f"[build] {engine_path} exists, skipping")
        return

    # Check for a pre-downloaded ONNX first
    if os.path.exists(onnx_path):
        print(f"[build] Using existing {onnx_path}", flush=True)
        build_engine_from_onnx(onnx_path, engine_path)
        return

    print("[build] Exporting UFLD (RESA) to ONNX ...", flush=True)
    import torch

    try:
        model = torch.hub.load("ZJULearning/RESA", "RESA_Net", pretrained=True)
    except Exception as e:
        print(f"[build] torch.hub failed: {e}", flush=True)
        print("[build] Place a pre-exported ufld.onnx in models/ and re-run.", flush=True)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    dummy = torch.zeros(1, 3, 288, 800, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"[build] ONNX saved to {onnx_path}", flush=True)
    build_engine_from_onnx(onnx_path, engine_path)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_fcn_engine()
    build_ufld_engine()
    print("[build] Done. Engines are in models/", flush=True)
