# Use an NVIDIA PyTorch base image compatible with Jetson Orin Nano on JetPack 6.x
FROM nvcr.io/nvidia/pytorch:24.05-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV and basic utilities
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the YOLOPv2 repository
RUN git clone https://github.com/CAIC-AD/YOLOPv2.git .

# Install YOLOPv2 requirements if any, and our streaming requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create weights directory
RUN mkdir -p data/weights

# Download a sample driving video for testing without a camera
RUN wget -O sample.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4

# Copy our custom streaming application
COPY app.py .

EXPOSE 5000

# Start the Flask streaming application
CMD ["python3", "app.py"]
