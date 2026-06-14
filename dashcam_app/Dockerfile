# Use the NVIDIA NGC PyTorch iGPU Container (includes TensorRT + PyTorch)
# JetPack 5.1.2 / CUDA 11.4 / cuDNN 8.6 / TensorRT 8.5
FROM nvcr.io/nvidia/pytorch:24.03-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

WORKDIR /app

# Install system dependencies for DeepStream, GStreamer, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    python3-opencv \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Flask, etc.)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install MMCV from OpenMMLab Jetson wheel index (pre-compiled for aarch64)
RUN pip3 install --no-cache-dir \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/jetson/index.html

# Install MMDetection and MMDetection3D
RUN pip3 install --no-cache-dir \
    mmdet==3.3.0 \
    mmdet3d==1.4.0

# Install MMDeploy for model conversion (PyTorch → ONNX → TensorRT).
# Build-time only; runtime uses the pre-compiled .engine file.
# For Deformable Attention TensorRT plugins, compile from:
# https://github.com/DerryHub/BEVFormer_tensorrt
RUN pip3 install --no-cache-dir mmdeploy==0.2.0

# Install MMDeploy TensorRT runtime if available on this platform.
# If unavailable, custom BEVFormer DeformableAttn plugins must be compiled manually
# from https://github.com/DerryHub/BEVFormer_tensorrt
RUN pip3 install --no-cache-dir mmdeploy-runtime-gpu \
    -f https://download.openmmlab.com/mmdeploy/jetson/index.html \
    2>/dev/null; true

# Install CuPy - optional GPU-accelerated post-processing.
# Pre-built wheels available for x86_64 (cupy-cuda11x, cupy-cuda12x).
# On Jetson aarch64, no pre-built wheel exists; building from source takes ~30min:
#   pip3 install cupy --no-build-isolation
# The ADAS postprocessor falls back to NumPy automatically if CuPy is unavailable.
RUN pip3 install --no-cache-dir cupy-cuda11x 2>/dev/null \
    || echo "[INFO] CuPy not installed; ADAS postprocessor will use NumPy fallback"

# Uninstall any pip opencv packages that conflict with Jetson native opencv
RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# Create directories
RUN mkdir -p recordings models

# Copy application files
COPY trt_runner.py .
COPY twinlite_detector.py .
COPY fcn_detector.py .
COPY bevformer_detector.py .
COPY deepstream_pipeline.py .
COPY adas_postprocessor.py .
COPY build_engines.py .
COPY dashcam.py .
COPY templates/ templates/
COPY install_service.sh .

EXPOSE 5001

# Start the dashcam application
CMD ["python3", "dashcam.py"]
