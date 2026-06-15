FROM nvcr.io/nvidia/pytorch:24.03-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    python3-opencv \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

RUN mkdir -p recordings models

COPY trt_runner.py .
COPY yolop_detector.py .
COPY dashcam.py .
COPY templates/ templates/

EXPOSE 5001

CMD ["python3", "dashcam.py"]
