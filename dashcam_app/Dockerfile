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

# Clone the YOLOPv2 repository for utils
RUN git clone https://github.com/CAIC-AD/YOLOPv2.git .

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create weights and recordings directories
RUN mkdir -p data/weights recordings

# Copy our custom dashcam application
COPY dashcam.py .
COPY templates/ templates/

EXPOSE 5001

# Start the Flask dashcam application
CMD ["python3", "dashcam.py"]
