# Use the community-maintained L4T PyTorch base image for JetPack 6.x
# This image contains PyTorch and TensorRT, avoiding the bloat of jetson-inference.
FROM dustynv/l4t-pytorch:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir ultralytics --extra-index-url https://pypi.org/simple

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
