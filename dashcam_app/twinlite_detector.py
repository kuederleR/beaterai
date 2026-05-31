import cv2
import numpy as np
import torch
import urllib.request
import os
import sys

# Ensure TwinLite can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TwinLite import TwinLiteNet

class TwinLiteDetector:
    def __init__(self, model_path="data/weights/best.pth"):
        self.model_path = model_path
        self.input_width = 640
        self.input_height = 360
        
        self.download_model_if_missing()
        
        # Initialize PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] TwinLiteNet using device: {self.device}")
        
        try:
            self.model = TwinLiteNet()
            self.model = torch.nn.DataParallel(self.model)
            
            # Use map_location to ensure it loads properly regardless of the device it was saved on
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            
            self.model = self.model.module
            if self.device.type == 'cuda':
                self.model = self.model.half()
            self.model = self.model.to(self.device)
            self.model.eval()
            print("[INFO] TwinLiteNet PyTorch Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load TwinLiteNet model: {e}")
            self.model = None

    def download_model_if_missing(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading TwinLiteNet PyTorch model to {self.model_path}...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = "https://raw.githubusercontent.com/chequanghuy/TwinLiteNet/main/pretrained/best.pth"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("[INFO] Download complete.")

    def preprocess(self, img):
        # Resize to input dimensions
        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize: TwinLiteNet uses standard ImageNet normalization
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # CHW -> add batch dimension
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            img_tensor = img_tensor.half()
        return img_tensor

    def detect(self, img):
        if self.model is None:
            return None, None
            
        h, w = img.shape[:2]
        tensor = self.preprocess(img)
        
        with torch.no_grad():
            da_out, ll_out = self.model(tensor)
            
        # da_out is [1, 2, 360, 640]
        _, da_predict = torch.max(da_out, 1)
        _, ll_predict = torch.max(ll_out, 1)
        
        da_mask = da_predict.squeeze().cpu().numpy().astype(np.uint8) # shape [360, 640]
        ll_mask = ll_predict.squeeze().cpu().numpy().astype(np.uint8) # shape [360, 640]
        
        # Resize masks back to original image size
        da_mask_resized = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ll_mask_resized = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return da_mask_resized, ll_mask_resized
