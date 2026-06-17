#!/bin/bash
set -e

. /opt/ros/jazzy/setup.sh
. /workspace/install/setup.sh

export PATH
export PYTHONPATH
export AMENT_PREFIX_PATH
export LD_LIBRARY_PATH

exec /opt/nvidia/nvidia_entrypoint.sh "$@"
