import cv2
import numpy as np
import torch
import torchvision
import urllib.request
import os
import time
import subprocess
import warnings

# --- TensorRT import (optional) ---
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    pass

# --- Helper functions copied/adapted from CAIC-AD/YOLOPv2 utils.py ---

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def split_for_trace_model(pred=None, anchor_grid=None):
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = pred[i].sigmoid()
        gr = _make_grid(nx, ny).to(pred[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
        z.append(y.view(bs, -1, 85))
    pred = torch.cat(z, 1)
    return pred

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    max_det = 300
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > 30000:
            x = x[x[:, 4].argsort(descending=True)[:30000]]
        c = x[:, 5:6] * 4096
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, [0, 2]].clamp_(0, img0_shape[1])
    coords[:, [1, 3]].clamp_(0, img0_shape[0])
    return coords


# --- TensorRT engine runner (uses PyTorch tensors for device memory, no pycuda) ---

class TrtRunner:
    def __init__(self, engine_path):
        self.engine = None
        self.context = None
        self.input_idx = None
        self.output_idxs = []
        self.input_shape = None
        self.output_shapes = []
        self.d_input = None
        self.d_outputs = []
        self.bindings = []

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                self.input_idx = i
                self.input_shape = shape
            else:
                self.output_idxs.append(i)
                self.output_shapes.append(shape)

    def infer(self, input_np):
        if self.d_input is None:
            self.d_input = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
            self.d_outputs = [torch.empty(s, dtype=torch.float32, device='cuda') for s in self.output_shapes]
            self.bindings = [self.d_input.data_ptr()] + [o.data_ptr() for o in self.d_outputs]

        # Copy input to GPU
        self.d_input.copy_(torch.from_numpy(np.ascontiguousarray(input_np)).cuda())

        # Run
        self.context.execute_v2(self.bindings)

        # Copy outputs back
        return [o.cpu().numpy() for o in self.d_outputs]


# --- YolopDetector wrapper class ---

class YolopDetector:
    def __init__(self, model_path="data/weights/yolopv2.pt"):
        self.model_path = model_path
        self.img_size = 640
        self.download_model_if_missing()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] YolopDetector using device: {self.device}", flush=True)

        self.half = self.device.type != 'cpu'
        self.model = None
        self.trt_runner = None

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("[INFO] Enabled cudnn.benchmark", flush=True)

        self._load_model()

    def _load_model(self):
        # Load TorchScript model (always needed for warmup + fallback)
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            if self.half:
                self.model = self.model.half()
            self.model.eval()
            if hasattr(torch.jit, 'optimize_for_inference'):
                try:
                    self.model = torch.jit.optimize_for_inference(self.model)
                except Exception:
                    pass
            print("[INFO] YOLOpv2 TorchScript model loaded.", flush=True)

            with torch.inference_mode():
                dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
                if self.half:
                    dummy = dummy.half()
                dummy = dummy.type_as(next(self.model.parameters()))
                self.model(dummy)
            print("[INFO] YOLOpv2 model warmed up.", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOpv2 model: {e}", flush=True)
            self.model = None
            return

        # --- Try to build/load TensorRT engine ---
        if not TRT_AVAILABLE or self.device.type != 'cuda':
            return

        engine_path = self.model_path.replace('.pt', '.trt')
        onnx_path = self.model_path.replace('.pt', '.onnx')

        if os.path.exists(engine_path):
            try:
                self.trt_runner = TrtRunner(engine_path)
                print("[INFO] Loaded cached TensorRT engine.", flush=True)
                return
            except Exception as e:
                print(f"[WARN] Failed to load TRT engine: {e}", flush=True)

        # Build TRT engine via trtexec (comes with every JetPack)
        print("[INFO] Building TensorRT engine via trtexec (may take several minutes)...", flush=True)
        try:
            self._export_onnx(onnx_path)
            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={engine_path}',
                '--fp16',
                '--workspace=1024',
                '--useCudaGraph',
                '--noDataTransfers',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"trtexec failed: {result.stderr[:500]}")
            self.trt_runner = TrtRunner(engine_path)
            print("[INFO] TensorRT engine built and loaded.", flush=True)
        except Exception as e:
            print(f"[WARN] TensorRT build failed, using PyTorch: {e}", flush=True)
            self.trt_runner = None

    def _export_onnx(self, onnx_path):
        dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
        if self.half:
            dummy = dummy.half()
        dummy = dummy.type_as(next(self.model.parameters()))
        torch.onnx.export(
            self.model, dummy, onnx_path,
            input_names=["input"],
            output_names=["pred", "anchor_grid", "seg", "ll"],
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"[INFO] ONNX exported to {onnx_path}", flush=True)

    def download_model_if_missing(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading YOLOpv2 model weights to {self.model_path}...", flush=True)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("[INFO] YOLOpv2 weights download complete.", flush=True)

    def detect(self, img):
        if self.model is None:
            return [], None, None

        h_orig, w_orig = img.shape[:2]

        # --- Preprocess (same for both paths) ---
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        if self.trt_runner is not None:
            # --- TensorRT path ---
            pred_np, anchor_grid_np, seg_np, ll_np = self.trt_runner.infer(img_chw)
            pred = [torch.from_numpy(p).to(self.device) for p in pred_np]
            anchor_grid = [torch.from_numpy(a).to(self.device) for a in anchor_grid_np]
            seg = torch.from_numpy(seg_np).to(self.device)
            ll = torch.from_numpy(ll_np).to(self.device)
        else:
            # --- PyTorch path ---
            img_tensor = torch.from_numpy(img_chw).to(self.device)
            if self.half:
                img_tensor = img_tensor.half()
            img_tensor = img_tensor.type_as(next(self.model.parameters()))
            with torch.inference_mode():
                [pred, anchor_grid], seg, ll = self.model(img_tensor)
                pred = list(pred)
                anchor_grid = list(anchor_grid)

        # --- Common post-processing ---
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=[2, 3, 4])

        # Scale boxes back to original image size
        det_boxes = []
        if len(pred) > 0 and pred[0] is not None and len(pred[0]) > 0:
            det = pred[0].clone()
            det[:, :4] = scale_coords((self.img_size, self.img_size), det[:, :4], (h_orig, w_orig)).round()
            for *xyxy, conf, cls in reversed(det):
                det_boxes.append({
                    "x1": float(xyxy[0].cpu().numpy()),
                    "y1": float(xyxy[1].cpu().numpy()),
                    "x2": float(xyxy[2].cpu().numpy()),
                    "y2": float(xyxy[3].cpu().numpy()),
                    "conf": float(conf.cpu().numpy()),
                    "class": int(cls.cpu().numpy())
                })

        # Segmentation masks
        _, da_predict_idx = torch.max(seg, 1)
        da_mask = da_predict_idx.squeeze().cpu().numpy().astype(np.uint8)
        ll_mask = torch.round(ll).squeeze().cpu().numpy().astype(np.uint8)

        da_mask = cv2.resize(da_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        return det_boxes, da_mask, ll_mask
