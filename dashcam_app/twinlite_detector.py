import cv2
import numpy as np
import onnxruntime
import urllib.request
import os

class TwinLiteDetector:
    def __init__(self, model_path="data/weights/twinlitenet.onnx"):
        self.model_path = model_path
        self.input_width = 640
        self.input_height = 360
        
        self.download_model_if_missing()
        
        # Initialize ONNX runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print("[INFO] TwinLiteNet Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load TwinLiteNet model: {e}")
            self.session = None

    def download_model_if_missing(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading TwinLiteNet ONNX model to {self.model_path}...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = "https://raw.githubusercontent.com/harrylal/TwinLiteNet-onnx-opencv-dnn/main/models/best.onnx"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("[INFO] Download complete.")

    def preprocess(self, img):
        # Resize to input dimensions
        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize: TwinLiteNet typically uses standard ImageNet normalization
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # CHW -> add batch dimension
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_tensor = np.expand_dims(img_chw, axis=0)
        return img_tensor

    def detect(self, img):
        if self.session is None:
            return None, None
            
        h, w = img.shape[:2]
        tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: tensor})
        
        # outputs[0] is da (drivable area), outputs[1] is ll (lane lines)
        da_output = outputs[0][0] # shape [2, 360, 640]
        ll_output = outputs[1][0] # shape [2, 360, 640]
        
        # Argmax to get class indices
        da_mask = np.argmax(da_output, axis=0).astype(np.uint8) # shape [360, 640]
        ll_mask = np.argmax(ll_output, axis=0).astype(np.uint8) # shape [360, 640]
        
        # Resize masks back to original image size using nearest neighbor
        da_mask_resized = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ll_mask_resized = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return da_mask_resized, ll_mask_resized
