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
    padx = (img1_shape[1] - img0_shape[1] * gain) / 2
    pady = (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= padx
    coords[:, [1, 3]] -= pady
    coords[:, :4] /= gain
    coords[:, [0, 2]].clamp_(0, img0_shape[1])
    coords[:, [1, 3]].clamp_(0, img0_shape[0])
    return coords


# --- TensorRT engine runner (PyTorch tensors for device memory, no pycuda) ---

class TrtRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = None
        self.output_shapes = []
        self.d_input = None
        self.d_outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                self.input_shape = shape
            else:
                self.output_shapes.append(shape)

    def infer(self, input_np):
        if self.d_input is None:
            self.d_input = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
            self.d_outputs = [torch.empty(s, dtype=torch.float32, device='cuda') for s in self.output_shapes]
            self.bindings = [self.d_input.data_ptr()] + [o.data_ptr() for o in self.d_outputs]
        self.d_input.copy_(torch.from_numpy(np.ascontiguousarray(input_np)).cuda())
        self.context.execute_v2(self.bindings)
        return [o.cpu().numpy() for o in self.d_outputs]


# YOLOPv2 anchors (YOLOv5-style, 3 scales × 3 anchors × 2 dims)
YOLOP_ANCHOR_GRID = (
    np.array([[[[[10, 13], [16, 30], [33, 23]]]]], dtype=np.float32),   # P3/8
    np.array([[[[[30, 61], [62, 45], [59, 119]]]]], dtype=np.float32),  # P4/16
    np.array([[[[[116, 90], [156, 198], [373, 326]]]]], dtype=np.float32),  # P5/32
)

# --- YolopDetector wrapper class ---

class YolopDetector:
    def __init__(self, model_path="data/weights/yolopv2.pt", trt_engine_path=None):
        self.model_path = model_path
        self.trt_engine_path = trt_engine_path
        self.img_size = 480
        self.download_model_if_missing()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] YolopDetector using device: {self.device}", flush=True)

        self.half = self.device.type != 'cpu'
        self.model_dtype = None
        self.model = None
        self.trt_runner = None

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("[INFO] Enabled cudnn.benchmark", flush=True)

        # Load pre-built TRT engine if available (INT8 or FP16)
        if trt_engine_path and os.path.exists(trt_engine_path):
            try:
                self.trt_runner = TrtRunner(trt_engine_path)
                self._anchor_grid = None
                print(f"[INFO] Loaded TensorRT engine from {trt_engine_path}", flush=True)
                return
            except Exception as e:
                print(f"[WARN] Failed to load TRT engine {trt_engine_path}: {e}", flush=True)
                print("[INFO] Falling back to PyTorch inference.", flush=True)

        self._load_pytorch()
        self._build_trt()

    def _load_pytorch(self):
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

            try:
                param = next(self.model.parameters())
                self.model_dtype = param.dtype
            except StopIteration:
                self.model_dtype = torch.float16 if self.half else torch.float32
            print(f"[INFO] Model parameter dtype: {self.model_dtype}", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Failed to load YOLOpv2 model: {e}", flush=True)
            self.model = None
            return

        # Warmup is optional — skip on failure rather than discarding the model
        self._anchor_grid = None
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, self.img_size, self.img_size, dtype=self.model_dtype).to(self.device)
                [_, anchor_grid], _, _ = self.model(dummy)
                self._anchor_grid = [ag.cpu().float() for ag in anchor_grid]
            print("[INFO] YOLOpv2 model warmed up. Ready for inference.", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] Model warmup failed (continuing without warmup): {e}", flush=True)

    def _build_trt(self):
        # Only bother on CUDA with TensorRT installed
        if not TRT_AVAILABLE or self.device.type != 'cuda' or self.model is None:
            print("[INFO] TensorRT not available, using PyTorch inference.", flush=True)
            return

        engine_path = self.model_path.replace('.pt', '.trt')

        # Load cached engine instantly
        if os.path.exists(engine_path):
            try:
                self.trt_runner = TrtRunner(engine_path)
                print("[INFO] Loaded cached TensorRT engine.", flush=True)
                return
            except Exception as e:
                print(f"[WARN] Failed to load cached TRT engine: {e}", flush=True)

        # Build engine (synchronous — inference waits for this to complete)
        try:
            print("[INFO] Exporting ONNX for TensorRT...", flush=True)
            onnx_path = self.model_path.replace('.pt', '.onnx')
            self._export_onnx(onnx_path)

            # Check trtexec availability
            if subprocess.run('which trtexec', shell=True, capture_output=True).returncode != 0:
                raise RuntimeError("trtexec not found on PATH — install via 'apt install nvidia-tensorrt'")

            print("[INFO] Building TensorRT engine via trtexec...", flush=True)
            cmd = [
                'stdbuf', '-oL',
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={engine_path}',
                # FP16 causes NaN outputs for lane/seg heads on Orin, stick with FP32
                # '--fp16',
                '--memPoolSize=workspace:1024',
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(f"[trtexec] {line.rstrip()}", flush=True)
            proc.wait(timeout=600)
            if proc.returncode != 0:
                raise RuntimeError(f"trtexec failed with code {proc.returncode}")
            self.trt_runner = TrtRunner(engine_path)
            print("[INFO] TensorRT engine ready. Switched to TRT inference.", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] TensorRT build failed, using PyTorch: {e}", flush=True)

    def _export_onnx(self, onnx_path):
        dummy = torch.zeros(1, 3, self.img_size, self.img_size, dtype=self.model_dtype).to(self.device)
        torch.onnx.export(
            self.model, dummy, onnx_path,
            input_names=["input"],
            output_names=["pred", "ag0", "ag1", "ag2", "seg", "ll"],
            opset_version=12,
            do_constant_folding=True,
        )
        # Post-process ONNX to remove SequenceConstruct ops (not supported by TRT)
        try:
            self._flatten_onnx_sequences(onnx_path)
        except Exception as e:
            print(f"[WARN] ONNX sequence flattening failed: {e}. TRT build may fail.", flush=True)

    def _flatten_onnx_sequences(self, onnx_path):
        import onnx
        from onnx import helper, TensorProto

        model = onnx.load(onnx_path)
        graph = model.graph

        # Log original outputs before flattening
        orig_out = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim] if o.type.tensor_type.shape else []) for o in graph.output]
        print(f"[INFO] ONNX original outputs: {orig_out}", flush=True)

        seq_outputs = {}
        graph_output_names = set(o.name for o in graph.output)
        for node in list(graph.node):
            if node.op_type == 'SequenceConstruct':
                if node.output[0] in graph_output_names:
                    seq_outputs[node.output[0]] = list(node.input)
                    graph.node.remove(node)
                # Internal SequenceConstruct nodes must stay — removing them
                # breaks the model's internal list operations (e.g. multi-scale
                # feature concatenation), causing TensorRT to produce NaN.

        original_outputs = list(graph.output)
        del graph.output[:]
        for output in original_outputs:
            if output.name in seq_outputs:
                for inp_name in seq_outputs[output.name]:
                    val_info = next(
                        (vi for vi in graph.value_info if vi.name == inp_name),
                        None
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

        # Run shape inference to propagate shapes for flattened outputs
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph

        onnx.save(model, onnx_path)

        # Log final output structure for use by TRT inference path
        final_out = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim] if o.type.tensor_type.shape else []) for o in graph.output]
        print(f"[INFO] ONNX final outputs: {final_out}", flush=True)
        self._trt_output_names = [o.name for o in graph.output]
        print(f"[INFO] ONNX sequences flattened: {seq_outputs}", flush=True)

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
        if self.model is None and self.trt_runner is None:
            return [], None, None

        h_orig, w_orig = img.shape[:2]

        # --- Preprocess (same for both paths) ---
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        trt = self.trt_runner
        if trt is not None:
            # --- TensorRT path ---
            try:
                out = trt.infer(img_chw)
                t0 = time.perf_counter()
                pred = [torch.from_numpy(out[i]).to(self.device) for i in range(3)]
                # Find seg and ll by shape (TRT may reorder bindings)
                seg = ll = None
                H = self.img_size
                for arr in out:
                    s = arr.shape
                    if len(s) == 4 and s[1] == 2 and s[2] == H and s[3] == H:
                        seg = torch.from_numpy(arr).to(self.device)
                    elif len(s) == 4 and s[1] == 1 and s[2] == H and s[3] == H:
                        ll = torch.from_numpy(arr).to(self.device)
                if seg is None or ll is None:
                    raise RuntimeError(f"Could not locate seg/ll in {len(out)} TRT outputs: "
                                       f"{[o.shape for o in out]}")
                ll_nan = torch.isnan(ll).sum().item()
                seg_nan = torch.isnan(seg).sum().item()
                if ll_nan > 0 or seg_nan > 0:
                    raise RuntimeError(f"TRT produced NaN (ll_nan={ll_nan}/{ll.numel()}, "
                                       f"seg_nan={seg_nan}/{seg.numel()})")
                print(f"[TRT] {len(out)} outputs, seg={seg.shape} ll={ll.shape} "
                      f"ll_range=[{ll.min().item():.4f}, {ll.max().item():.4f}] "
                      f"ll_nan={ll_nan}", flush=True)
                # Use cached anchor_grid (TRT may fold out these constants)
                if self._anchor_grid is not None:
                    anchor_grid = [ag.to(self.device) for ag in self._anchor_grid]
                else:
                    anchor_grid = [torch.from_numpy(ag).to(self.device) for ag in YOLOP_ANCHOR_GRID]
                pred = split_for_trace_model(pred, anchor_grid)
                t1 = time.perf_counter()
                print(f"[TIMING] TRT path: {((t1 - t0) * 1000):.1f}ms", flush=True)
            except Exception as e:
                print(f"[WARN] TRT inference failed, falling back to PyTorch: {e}", flush=True)
                # Delete bad engine so next restart rebuilds without FP16
                engine_path = self.model_path.replace('.pt', '.trt')
                if os.path.exists(engine_path):
                    os.remove(engine_path)
                    print("[INFO] Deleted bad TRT engine, will rebuild on next startup.", flush=True)
                self.trt_runner = None
                trt = None

        if trt is None:
            # --- PyTorch path ---
            t0 = time.perf_counter()
            img_tensor = torch.from_numpy(img_chw.astype(np.float32)).to(self.device, dtype=self.model_dtype)
            with torch.inference_mode():
                [pred, anchor_grid], seg, ll = self.model(img_tensor)
                pred = list(pred)
                anchor_grid = list(anchor_grid)
            pred = split_for_trace_model(pred, anchor_grid)
            t1 = time.perf_counter()
            print(f"[TIMING] PyTorch path: {((t1 - t0) * 1000):.1f}ms", flush=True)
            print(f"[PYTORCH] seg={seg.shape} ll={ll.shape} "
                  f"ll_range=[{ll.min().item():.4f}, {ll.max().item():.4f}]", flush=True)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=[2, 3, 4])

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

        seg = torch.nan_to_num(seg, 0)
        _, da_predict_idx = torch.max(seg, 1)
        da_mask = da_predict_idx.squeeze().cpu().numpy().astype(np.uint8)
        ll = torch.nan_to_num(ll, 0).clamp(0, 1)
        ll_mask = torch.round(ll).squeeze().cpu().numpy().astype(np.uint8)

        da_mask = cv2.resize(da_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        return det_boxes, da_mask, ll_mask
