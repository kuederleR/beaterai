# Use an NVIDIA PyTorch base image compatible with Jetson Orin Nano on JetPack 6.x
FROM nvcr.io/nvidia/pytorch:24.05-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: OpenCV support, ffmpeg, GStreamer plugins
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Nuke the broken/headless system cv2 to prevent namespace conflicts with pip
RUN rm -rf /usr/lib/python3.10/dist-packages/cv2 \
           /usr/local/lib/python3.10/dist-packages/cv2

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Attempt to install Jetson ONNXRuntime GPU. If it fails, fallback to CPU version.
RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu122/ onnxruntime-gpu || pip3 install --no-cache-dir onnxruntime
# Create weights and recordings directories
RUN mkdir -p data/weights recordings

# Copy our custom dashcam application
COPY yolop_detector.py .
COPY dashcam.py .
COPY templates/ templates/
COPY install_service.sh .

EXPOSE 5001

# Start the Flask dashcam application
CMD ["python3", "dashcam.py"]
