import cv2
import numpy as np
import onnxruntime
import scipy.special
import urllib.request
import os

class UFLDDetector:
    def __init__(self, model_path="data/weights/ufld_culane_288x800.onnx"):
        self.model_path = model_path
        self.input_width = 800
        self.input_height = 288
        self.griding_num = 200
        self.cls_num_per_lane = 18
        # CULane row anchors
        self.row_anchors = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
        
        self.download_model_if_missing()
        
        # Initialize ONNX runtime (will attempt CUDA first, fallback to CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print("[INFO] UFLD Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load UFLD model: {e}")
            self.session = None
        
    def download_model_if_missing(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading UFLD ONNX model to {self.model_path}...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # This is an ~11MB ONNX model from the popular community inference repo
            url = "https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference/raw/main/models/ultra_fast_lane_detection_culane_288x800.onnx"
            # Setting a user agent in case GitHub blocks vanilla urllib
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("[INFO] Download complete.")

    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_norm = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std
        
        # HWC to CHW -> Add batch dimension
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_tensor = np.expand_dims(img_chw, axis=0)
        return img_tensor

    def postprocess(self, output, original_width, original_height):
        # output[0] shape is [201, 18, 4] for CULane
        output = output[0]
        
        # Softmax over the 200 grid columns (ignoring the 201st "no lane" class)
        prob = scipy.special.softmax(output[:-1, :, :], axis=0) # [200, 18, 4]
        
        # Calculate expected location
        idx = np.arange(self.griding_num) + 1
        idx = idx.reshape(-1, 1, 1) # [200, 1, 1]
        loc = np.sum(prob * idx, axis=0) # [18, 4]
        
        # Check if "no lane" class has the highest probability
        out_j = np.argmax(output, axis=0) # [18, 4]
        
        lanes = []
        for i in range(4): # 4 lane positions
            lane = []
            for j in range(self.cls_num_per_lane): # 18 anchors
                if out_j[j, i] == self.griding_num: # Not detected
                    continue
                
                # Transform coordinate back to original image
                x = (loc[j, i] / self.griding_num) * original_width
                y = (self.row_anchors[j] / self.input_height) * original_height
                
                lane.append((int(x), int(y)))
            lanes.append(lane)
            
        return lanes

    def detect(self, img):
        if self.session is None:
            return [[], [], [], []]
            
        h, w = img.shape[:2]
        tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: tensor})
        
        lanes = self.postprocess(outputs[0], w, h)
        return lanes
