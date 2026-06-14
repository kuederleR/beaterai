# Edge ADAS: Real-Time BEV Deployment Architecture

**Target Hardware:** NVIDIA Jetson (Orin Nano / NX / AGX)  
**Constraint:** Zero custom training; utilizing 100% open-source pre-trained weights.

---

## 1. Core AI Architecture: BEVFormer-Tiny

To achieve Bird's-Eye View (BEV) transformation without a dedicated depth-estimation network, the architecture relies on **Spatio-Temporal Transformers**. We select **BEVFormer-Tiny** because it leverages deterministic camera parameters (intrinsics/extrinsics) to generate the top-down view, eliminating the need to retrain projection layers.

### Where to Source the Models & Weights
* **Framework:** [OpenMMLab / MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* **Model:** BEVFormer-Tiny (nuScenes config)
* **Pre-trained Weights:** Available in the MMDetection3D Model Zoo. 
    * *Note: Download the weights trained on the `nuScenes` dataset. nuScenes contains annotations for 3D bounding boxes and HD map lanes, which activate both the tracking and lane segmentation heads.*
* **TensorRT Deployment Repo:** [DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) (Crucial for compiling BEVFormer's custom attention layers into a Jetson-compatible TensorRT engine).

---

## 2. The End-to-End Inference Pipeline

### Phase 1: Video Ingestion & Zero-Copy Memory
* **Tool:** NVIDIA DeepStream SDK / GStreamer
* **Process:** The dashcam feed is captured and decoded directly into **NVMM (NVIDIA Memory Management)** buffers.
* **Why this matters:** On a Jetson, the CPU and GPU share physical RAM. Using NVMM prevents the system from copying the 1080p video frame from the CPU to the GPU, saving massive amounts of latency.

### Phase 2: The BEVFormer Forward Pass (TensorRT Engine)
* **Backbone (ResNet-50 / ResNet-18):** Extracts 2D multi-scale semantic features from the front-facing camera frame.
* **Spatial Cross-Attention (The View Transformer):**
    * A 100x100 grid is initialized in the BEV plane.
    * Using the camera's fixed extrinsic and intrinsic matrices, the network projects 3D reference points from the BEV grid back to the 2D image plane to "query" the pixel features.
* **Multi-Task Decoders:**
    * **Detection Head:** Outputs CenterPoint 3D bounding boxes (X, Y, Z, width, length, height, heading angle).
    * **Map Head:** Outputs a semantic segmentation mask in the BEV plane representing the ego-lane boundaries.

### Phase 3: Post-Processing & Ego-Calculations (CUDA / CuPy)
Because the network weights are frozen, raw outputs must be translated into your ADAS metrics using classical math.

1.  **Lane Curve Fitting:**
    * Extract the X, Y coordinates of the highest-probability pixels from the Map Head's lane mask.
    * Use CuPy (GPU-accelerated Python) to fit a **3rd-degree polynomial** (`y = ax^3 + bx^2 + cx + d`) to the left and right lane boundaries.
2.  **Ego-Vehicle Offset:**
    * The BEV grid coordinate (0,0) represents the exact location of the camera.
    * Evaluate your left and right polynomial equations at `y = 0` (your vehicle's current lateral axis).
    * The midpoint between the left and right lane coordinates compared to (0,0) yields your precise lateral deviation in meters.
3.  **Heading Error:**
    * Calculate the derivative of the lane polynomial at `y = 0` to find the tangent line, which represents your vehicle's heading angle relative to the lane direction.

---

## 3. The Software Stack

| Component | Tool / Library | Function |
| :--- | :--- | :--- |
| **Model Conversion** | **MMDeploy** | Safely converts PyTorch weights and complex Transformer attention layers into an optimized `.onnx` graph. |
| **Edge Compilation** | **NVIDIA TensorRT** | Compiles the `.onnx` file directly on the Jetson. **Must use INT8 or FP16 quantization** to achieve real-time FPS on the Orin Nano. |
| **Camera Pipeline** | **DeepStream** | Handles hardware-accelerated video decoding and feeds frames to the TensorRT engine. |
| **Math & Vectors** | **CuPy / OpenCV** | Runs the polynomial curve fitting and ego-offset math entirely on the Jetson's GPU to prevent CPU bottlenecks. |

---

## 4. Critical Deployment Warnings

### 1. The Extrinsics Constraint (The "NuScenes Match" Rule)
Because you are using pre-trained weights without fine-tuning, **the network assumes your camera is physically mounted in the exact same position as the nuScenes dataset collection vehicles.**
* The nuScenes front camera is mounted high, centered, and looking straight ahead (minimal pitch).
* If your dashcam is mounted lower, off-center, or tilted, the pre-trained attention layers will project the BEV grid incorrectly. Cars will appear stretched, and lanes will curve the wrong way.
* **The Fix:** You must calculate your physical dashcam's extrinsic matrix (translation and rotation relative to the car's center) and overwrite the nuScenes default extrinsic matrix in the model's configuration file before running inference. 

### 2. TensorRT Plugin Requirements
Standard TensorRT does not natively support the Deformable Attention operations used in BEVFormer. You **must** compile the custom TensorRT plugins provided by MMDeploy or the `BEVFormer_tensorrt` repository to allow the network to execute on the Jetson's Tensor Cores.
