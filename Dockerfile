# Use an NVIDIA PyTorch base image compatible with Jetson Orin Nano on JetPack 6.x
FROM nvcr.io/nvidia/pytorch:24.05-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: OpenCV support, ffmpeg, GStreamer plugins for mp4 decoding
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

# Clone the YOLOPv2 repository
RUN git clone https://github.com/CAIC-AD/YOLOPv2.git .

# Install streaming requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create weights directory
RUN mkdir -p data/weights

# Download a proper dashcam driving video (Udacity self-driving car nanodegree project video)
# This is a highly reliable GitHub raw link of actual forward-facing highway driving
RUN curl -L -o sample.mp4 \
    "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/project_video.mp4" \
    && ls -lh sample.mp4 \
    && file sample.mp4

# Copy our custom streaming application
COPY app.py .

EXPOSE 5000

# Start the Flask streaming application
CMD ["python3", "app.py"]
