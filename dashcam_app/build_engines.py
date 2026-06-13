"""
Build TensorRT engines for TwinLiteNet.

Can run inside the Docker container or on the host Jetson.

Inside container (engines persist via ./models volume):
    docker compose run --rm dashcam python3 build_engines.py

Or on the host Jetson directly:
    python3 build_engines.py

Requirements inside container (already satisfied):
    - torch, torchvision  (nvcr.io base image)
    - tensorrt            (nvcr.io base image)
    - internet for model downloads

Output:
    models/twinlite.engine
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

def _load_fcn_model():
    """Try every available method to load FCN-ResNet18 with Cityscapes weights."""
    import torch
    import torchvision

    # Method 1: torchvision segmentation module (desktop builds)
    import_paths = [
        lambda: torchvision.models.segmentation.fcn_resnet18,
        lambda: torchvision.models.segmentation.fcn.fcn_resnet18,
    ]
    fn = None
    for path in import_paths:
        try:
            fn = path()
            break
        except (AttributeError, ImportError):
            continue

    if fn is not None:
        weight_args = [
            {"weights": torchvision.models.segmentation.FCN_ResNet18_Weights.CITYSCAPES_512x256},
            {"weights": "CITYSCAPES_512x256"},
            {"pretrained": True},
        ]
        for kwargs in weight_args:
            try:
                return fn(**kwargs)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue

    # Method 2: manual construction with direct weight download
    # (works when torchvision's segmentation module is not built, e.g. Jetson)
    print("[build] torchvision segmentation not available, building model manually...", flush=True)
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models.resnet import resnet18
        from torchvision.models._utils import IntermediateLayerGetter

        class FCNHead(nn.Sequential):
            def __init__(self, in_channels, channels):
                super().__init__(
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv2d(channels, channels, 1),
                )

        class FCNWrapper(nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier

            def forward(self, x):
                input_shape = x.shape[-2:]
                features = self.backbone(x)
                x = self.classifier(features["out"])
                x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
                return {"out": x}

        backbone = resnet18(weights=None)
        backbone = IntermediateLayerGetter(backbone, {"layer4": "out"})
        classifier = FCNHead(512, 19)
        model = FCNWrapper(backbone, classifier)

        url = "https://download.pytorch.org/models/fcn_resnet18_cityscapes-2e0a3c0c.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict, strict=False)
        return model
    except Exception as e:
        print(f"[build] Manual construction failed: {e}", flush=True)
        return None


def build_fcn_engine():
    engine_path = "models/fcn_resnet18.engine"
    onnx_path = "models/fcn_resnet18.onnx"

    if os.path.exists(engine_path):
        print(f"[build] {engine_path} exists, skipping")
        return

    print("[build] Loading FCN-ResNet18 model ...", flush=True)
    model = _load_fcn_model()
    if model is None:
        print("[build] Could not load FCN-ResNet18 model", flush=True)
        print("[build] Place a pre-exported fcn_resnet18.onnx in models/ and re-run.", flush=True)
        sys.exit(1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    print("[build] Exporting FCN-ResNet18 to ONNX ...", flush=True)
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
# TwinLiteNet
# ---------------------------------------------------------------------------

TWINLITE_ONNX_URL = (
    "https://raw.githubusercontent.com/harrylal/"
    "TwinLiteNet-onnxruntime/main/models/best.onnx"
)

def build_twinlite_engine():
    engine_path = "models/twinlite.engine"
    onnx_path = "models/twinlite.onnx"

    if os.path.exists(engine_path):
        print(f"[build] {engine_path} exists, skipping")
        return

    if os.path.exists(onnx_path):
        print(f"[build] Using existing {onnx_path}", flush=True)
    else:
        print("[build] Downloading TwinLiteNet ONNX ...", flush=True)
        import urllib.request
        try:
            urllib.request.urlretrieve(TWINLITE_ONNX_URL, onnx_path)
            print(f"[build] ONNX saved to {onnx_path}", flush=True)
        except Exception as e:
            print(f"[build] Download failed: {e}", flush=True)
            print("[build] Place twinlite.onnx in models/ and re-run.", flush=True)
            sys.exit(1)

    build_engine_from_onnx(onnx_path, engine_path)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_twinlite_engine()
    print("[build] Done. Engines are in models/", flush=True)
