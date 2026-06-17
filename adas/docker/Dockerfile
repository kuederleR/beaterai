ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-colcon-common-extensions \
    v4l-utils \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-launch-ros \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    huggingface-hub \
    opencv-python-headless

# cv_bridge compiled against NumPy 1.x; force-downgrade
RUN pip3 install --no-cache-dir --force-reinstall 'numpy==1.26.4'

WORKDIR /workspace

COPY adas_ws/src /workspace/src
COPY test_diagnostic.py /workspace/test_diagnostic.py

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc && \
    echo "source /workspace/install/setup.bash" >> /etc/bash.bashrc

CMD ["ros2", "launch", "adas_launch", "adas_pipeline.launch.py"]
