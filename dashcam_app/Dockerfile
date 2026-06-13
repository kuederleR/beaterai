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
    cmake \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build jetson-inference from source
ENV BUILD_INTERACTIVE=0
RUN git clone --recursive https://github.com/dusty-nv/jetson-inference /opt/jetson-inference && \
    cd /opt/jetson-inference && \
    sed -i 's/set(PYTHON_BINDING_VERSIONS.*/set(PYTHON_BINDING_VERSIONS 3.10)/g' python/CMakeLists.txt utils/python/CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j2 && \
    make install && \
    ldconfig

WORKDIR /app

# Nuke the broken/headless system cv2 to prevent namespace conflicts with pip
RUN rm -rf /usr/lib/python3.10/dist-packages/cv2 \
           /usr/local/lib/python3.10/dist-packages/cv2

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create model weights and recordings directories
RUN mkdir -p recordings models

# Copy our custom dashcam application
COPY trt_runner.py .
COPY fcn_detector.py .
COPY ufld_detector.py .
COPY build_engines.py .
COPY dashcam.py .
COPY templates/ templates/
COPY install_service.sh .

EXPOSE 5001

# Start the Flask dashcam application
CMD ["python3", "dashcam.py"]
