# Use the NVIDIA NGC PyTorch iGPU Container (includes TensorRT + PyTorch)
FROM nvcr.io/nvidia/pytorch:24.03-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for DeepStream, GStreamer, OpenCV
RUN apt-get update && apt-get install -y \
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

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install MMDetection3D and MMCV for BEVFormer model zoo access
RUN pip3 install --no-cache-dir \
    mmcv>=2.0.0 \
    mmdet>=3.0.0 \
    mmdet3d>=1.4.0 \
    mmdeploy>=1.3.0

# Install CuPy for GPU-accelerated post-processing (Jetson CUDA 11.4)
RUN pip3 install --no-cache-dir cupy-cuda11x

# MMDeploy ONNX-Runtime and TensorRT plugins for Deformable Attention
RUN pip3 install --no-cache-dir mmdeploy-runtime-gpu

# Uninstall any pip opencv packages that conflict with Jetson native opencv
RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# Create directories
RUN mkdir -p recordings models

# Copy application files
COPY trt_runner.py .
COPY bevformer_detector.py .
COPY deepstream_pipeline.py .
COPY adas_postprocessor.py .
COPY dashcam.py .
COPY templates/ templates/
COPY install_service.sh .

EXPOSE 5001

# Start the dashcam application
CMD ["python3", "dashcam.py"]
