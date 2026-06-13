"""
Build TensorRT engines for FCN-ResNet18 and UFLD models.

Run once on the Jetson before starting the dashcam app:
    python3 build_engines.py

This exports torchvision FCN-ResNet18 and a
UFLD model to ONNX, then calls trtexec to build
TensorRT engines saved to models/fcn_resnet18.engine
and models/ufld.engine.
"""

import os
import subprocess
import sys

os.makedirs("models", exist_ok=True)


def build_fcn_engine():
    engine_path = "models/fcn_resnet18.engine"
    onnx_path = "models/fcn_resnet18.onnx"

    if os.path.exists(engine_path):
        print(f"[build] {engine_path} already exists, skipping")
        return

    print("[build] Exporting FCN-ResNet18 to ONNX...", flush=True)
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
        print("[build] Failed to load FCN-ResNet18 model", flush=True)
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

    print("[build] Building TensorRT engine via trtexec...", flush=True)
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--memPoolSize=workspace:1024",
    ]
    subprocess.run(cmd, check=True)
    print(f"[build] TensorRT engine saved to {engine_path}", flush=True)


def build_ufld_engine():
    engine_path = "models/ufld.engine"
    onnx_path = "models/ufld.onnx"

    if os.path.exists(engine_path):
        print(f"[build] {engine_path} already exists, skipping")
        return

    print("[build] Exporting UFLD to ONNX...", flush=True)
    import torch

    try:
        model = torch.hub.load("ZJULearning/RESA", "RESA_Net", pretrained=True)
    except Exception:
        print("[build] Could not load UFLD model from torch hub.", flush=True)
        print("[build] Download a pre-exported ONNX or build manually.", flush=True)
        print("[build] See https://github.com/ZJULearning/RESA for model export.", flush=True)
        return

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

    print("[build] Building TensorRT engine via trtexec...", flush=True)
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--memPoolSize=workspace:1024",
    ]
    subprocess.run(cmd, check=True)
    print(f"[build] TensorRT engine saved to {engine_path}", flush=True)


if __name__ == "__main__":
    build_fcn_engine()
    build_ufld_engine()
    print("[build] Done. Engines are in models/", flush=True)
