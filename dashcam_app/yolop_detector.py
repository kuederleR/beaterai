import cv2
import numpy as np
import torch
import torchvision
import urllib.request
import os
import time

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
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, 85))
    pred = torch.cat(z, 1)
    return pred

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output

def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


# --- YolopDetector wrapper class ---

class YolopDetector:
    def __init__(self, model_path="data/weights/yolopv2.pt"):
        self.model_path = model_path
        self.img_size = 640
        self.download_model_if_missing()
        
        # Initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] YolopDetector using device: {self.device}", flush=True)
        
        self.half = self.device.type != 'cpu'
        
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            if self.half:
                self.model = self.model.half()
            self.model.eval()
            print("[INFO] YOLOpv2 TorchScript model loaded successfully.", flush=True)
            
            # Warm up
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
                if self.half:
                    dummy = dummy.half()
                dummy = dummy.type_as(next(self.model.parameters()))
                self.model(dummy)
            print("[INFO] YOLOpv2 model warmed up.", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOpv2 model: {e}", flush=True)
            self.model = None

    def download_model_if_missing(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading YOLOpv2 model weights to {self.model_path}...", flush=True)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # Official release URL
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
        
        # 1. Preprocess: direct resize to model input size (no letterbox padding).
        #    This avoids wasting ~37% of inference compute on black padding rows
        #    while maintaining the same pixel count for the model.
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert image to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        
        if self.half:
            img_tensor = img_tensor.half()
            
        img_tensor = img_tensor.type_as(next(self.model.parameters()))
            
        # 2. Model Inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img_tensor)
            
            # Post-processing object detection
            pred = split_for_trace_model(pred, anchor_grid)
            # BDD100K class indices for vehicles: car (2), bus (3), truck (4)
            pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=[2, 3, 4])
            
        # Scale bounding boxes back to original image size
        det_boxes = []
        if len(pred) > 0 and pred[0] is not None:
            det = pred[0].clone()
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], (h_orig, w_orig)).round()
                for *xyxy, conf, cls in reversed(det):
                    det_boxes.append({
                        "x1": float(xyxy[0].cpu().numpy()),
                        "y1": float(xyxy[1].cpu().numpy()),
                        "x2": float(xyxy[2].cpu().numpy()),
                        "y2": float(xyxy[3].cpu().numpy()),
                        "conf": float(conf.cpu().numpy()),
                        "class": int(cls.cpu().numpy())
                    })
                    
        # 3. Post-process segmentation masks
        # Process drivable area (channels=2: background, road) -> get argmax
        _, da_predict_idx = torch.max(seg, 1)
        da_mask_model = da_predict_idx.squeeze().cpu().numpy().astype(np.uint8)
        
        # Process lane line area (channels=1) -> round to get binary mask
        ll_mask_model = torch.round(ll).squeeze().cpu().numpy().astype(np.uint8)
        
        # Resize masks back to original image size
        da_mask = cv2.resize(da_mask_model, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask_model, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
        return det_boxes, da_mask, ll_mask
