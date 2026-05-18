import sys
import os
import time
import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template_string

# Append current directory to path so we can import YOLOPv2 utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import select_device, scale_coords, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, plot_one_box

app = Flask(__name__)

# --- Configuration ---
WEIGHTS_PATH = 'data/weights/yolopv2.pt'
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', '0')
try:
    # Use integer for local webcam
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass

DEVICE_STR = os.environ.get('DEVICE', '0')
IMG_SIZE = 640
CONF_THRES = 0.3
IOU_THRES = 0.45

# --- Model Initialization ---
print(f"Loading YOLOPv2 model from {WEIGHTS_PATH}...")
if not os.path.exists(WEIGHTS_PATH):
    print("Weights not found, downloading...")
    os.makedirs('data/weights', exist_ok=True)
    os.system(f"wget -O {WEIGHTS_PATH} https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt")

device = select_device(DEVICE_STR)
half = device.type != 'cpu'

# Load model
model = torch.jit.load(WEIGHTS_PATH)
model = model.to(device)
if half:
    model.half()
model.eval()

# Dummy run to warmup the model
print("Warming up model...")
dummy_input = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device).type_as(next(model.parameters()))
with torch.no_grad():
    model(dummy_input)

# --- Utilities ---
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    print("Started generating frames...")
    while True:
        success, frame = cap.read()
        if not success:
            # Restart video if it's a file, otherwise we break
            if isinstance(VIDEO_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        im0 = frame.copy()
        h, w = im0.shape[:2]
        
        # Preprocess
        img, ratio, pad = letterbox(im0, IMG_SIZE, stride=32, auto=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = model(img_tensor)
            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)
            
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

        # Draw segmentations
        color_area = np.zeros((img_tensor.shape[2], img_tensor.shape[3], 3), dtype=np.uint8)
        
        da_mask = da_seg_mask[0] # assuming batch size 1
        ll_mask = ll_seg_mask[0]
        
        color_area[da_mask] = [0, 255, 0] # Drivable area: Green
        color_area[ll_mask] = [255, 0, 0] # Lane lines: Blue
        
        # Unpad and resize masks to original image size
        dw, dh = pad
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Crop out the padding
        color_area_cropped = color_area[top:IMG_SIZE-bottom, left:IMG_SIZE-right]
        # Resize to original resolution
        color_area_resized = cv2.resize(color_area_cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Blend segmentation with original image
        overlay = im0.copy()
        mask_indices = np.any(color_area_resized != [0, 0, 0], axis=-1)
        overlay[mask_indices] = color_area_resized[mask_indices]
        im0 = cv2.addWeighted(overlay, 0.5, im0, 0.5, 0)

        # Draw detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=3)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', im0)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string('''
    <html>
        <head>
            <title>YOLOPv2 Stream - Jetson Orin Nano</title>
            <style>
                body { background-color: #1a1a1a; color: white; display: flex; flex-direction: column; align-items: center; font-family: 'Inter', sans-serif; }
                h1 { margin-top: 20px; font-weight: 300; }
                .video-container { 
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #2a2a2a;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
                }
                img { 
                    max-width: 100%; 
                    border-radius: 8px; 
                }
            </style>
        </head>
        <body>
            <h1>YOLOPv2 Live Inference</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}">
            </div>
        </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
