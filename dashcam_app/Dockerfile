# Use the official jetson-inference base image for JetPack 6.x
# This image already contains PyTorch, OpenCV, GStreamer, and a fully compiled jetson-inference
FROM dustynv/jetson-inference:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

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
