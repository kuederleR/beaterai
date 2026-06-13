import cv2
import numpy as np
import os

from trt_runner import TrtRunner, TRT_AVAILABLE

CITYSCAPES_CLASSES = 19
ROAD_CLASS_ID = 0


class FCNResNetDetector:
    def __init__(self, engine_path="models/fcn_resnet18.engine"):
        self.input_height = 256
        self.input_width = 512

        self.trt_runner = None
        self.model = None
        self.device = None
        self.model_dtype = np.float32

        if TRT_AVAILABLE and os.path.exists(engine_path):
            try:
                print(f"[FCN] Loading TensorRT engine from {engine_path}", flush=True)
                self.trt_runner = TrtRunner(engine_path)
                print(f"[FCN] TensorRT engine ready.", flush=True)
            except Exception as e:
                print(f"[FCN] Failed to load TRT engine: {e}", flush=True)
        elif not TRT_AVAILABLE:
            print(f"[FCN] TensorRT not available. Loading PyTorch fallback.", flush=True)
            self._load_pytorch_fallback()
        else:
            print(f"[FCN] Engine file {engine_path} not found. Loading PyTorch fallback.", flush=True)
            print(f"[FCN] Run python3 build_engines.py to build it, or place a pre-built engine.", flush=True)
            self._load_pytorch_fallback()

    def _load_pytorch_fallback(self):
        try:
            import torch
            import torchvision
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self._try_load_fcn(torch, torchvision)
            if model is None:
                model = self._build_fcn_manual(torch)
            if model is not None:
                self.model = model.eval().to(self.device)
                print(f"[FCN] PyTorch fallback loaded (device={self.device})", flush=True)
            else:
                raise ImportError("Could not load FCN-ResNet18")
        except Exception as e:
            print(f"[FCN] PyTorch fallback failed: {e}", flush=True)
            self.model = None

    def _try_load_fcn(self, torch, torchvision):
        """Try every known API path to load FCN-ResNet18 from torchvision."""
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
        if fn is None:
            return None

        weight_args = [
            {"weights": torchvision.models.segmentation.FCN_ResNet18_Weights.CITYSCAPES_512x256},
            {"weights": "CITYSCAPES_512x256"},
            {"pretrained": True},
            {"pretrained": False},
        ]
        for kwargs in weight_args:
            try:
                m = fn(**kwargs)
                if kwargs.get("pretrained", False):
                    return m
                if kwargs.get("weights") is not None:
                    return m
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        return None

    def _build_fcn_manual(self, torch):
        """Construct FCN-ResNet18 manually using only basic torchvision blocks
        and download Cityscapes weights from the PyTorch model zoo directly.
        This works even when torchvision's segmentation module is not built."""
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
            print(f"[FCN] Manual construction failed: {e}", flush=True)
            return None

    def detect(self, img):
        if self.model is None and self.trt_runner is None:
            return None

        h_orig, w_orig = img.shape[:2]

        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        if self.trt_runner is not None:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_norm - mean) / std
            img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

            out = self.trt_runner.infer(img_chw)
            scores = out[0][0]
        else:
            import torch
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
            img_tensor = torch.from_numpy(img_chw).to(self.device)
            img_tensor = (img_tensor - mean) / std
            with torch.inference_mode():
                out = self.model(img_tensor)['out']
            scores = out[0].cpu().numpy()

        pred_class = np.argmax(scores, axis=0).astype(np.uint8)
        da_mask = (pred_class == ROAD_CLASS_ID).astype(np.uint8)
        da_mask = cv2.resize(da_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        return da_mask
