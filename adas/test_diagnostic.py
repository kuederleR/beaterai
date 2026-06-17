#!/usr/bin/env python3
"""ADAS setup diagnostic script.
Run this INSIDE the container to validate the deployment environment.

Usage:
  docker compose run --rm adas-pipeline-jetson python3 /workspace/test_diagnostic.py
"""

import importlib
import os
import subprocess
import sys
import time


def check(description, condition, hint=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if not condition and hint:
        print(f"         Hint: {hint}")
    return condition


def check_system():
    print("\n--- System Checks ---")
    results = []

    results.append(check(
        "Running on aarch64",
        os.uname().machine in ("aarch64", "arm64"),
    ))

    try:
        nv = open("/proc/driver/nvidia/version", "r").read().strip()
        results.append(check("NVIDIA driver found", True, ""))
        print(f"         {nv.split(chr(10))[0]}")
    except FileNotFoundError:
        results.append(check("NVIDIA driver found", False,
                             "Missing nvidia-container-toolkit on host"))

    dev = os.path.exists("/dev/video0")
    results.append(check("/dev/video0 exists", dev,
                         "Plug in USB camera or check device path"))

    if dev:
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "--list-formats-ext", "-d", "/dev/video0"],
                stderr=subprocess.STDOUT).decode()
            lines = out.split("\n")[:6]
            for line in lines:
                if line.strip():
                    print(f"         {line.strip()}")
        except Exception:
            pass

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            timeout=10).decode().strip()
        results.append(check("nvidia-smi responds", True))
        print(f"         GPU: {smi}")
    except Exception:
        results.append(check("nvidia-smi responds", False,
                             "GPU may not be accessible from container"))

    return all(results)


def call_python(code):
    try:
        exec(code, {})
        return True
    except Exception as e:
        print(f"         Error: {e}")
        return False


def check_python_deps():
    print("\n--- Python Dependencies ---")
    results = []

    for modname in ("numpy", "cv2", "torch", "transformers"):
        try:
            mod = importlib.import_module(modname)
            ver = getattr(mod, "__version__", "unknown")
            results.append(check(f"{modname} (v{ver})", True))
        except ImportError:
            results.append(check(f"{modname}", False,
                                 f"pip install {modname}"))

    try:
        import torch
        cuda = torch.cuda.is_available()
        results.append(check("torch.cuda available", cuda,
                             "Use Jetson PyTorch from NVIDIA index"))
        if cuda:
            device = torch.cuda.get_device_name(0)
            print(f"         Device: {device}")
    except Exception:
        pass

    return all(results)


def test_dummy_depth():
    print("\n--- Depth Model Smoke Test ---")
    try:
        import numpy as np
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_name = "depth-anything/Depth-Anything-V3-Small"
        print(f"  Downloading {model_name} ...")
        t0 = time.time()

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        t1 = time.time()
        print(f"  Model loaded in {t1-t0:.1f}s on {device}")

        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        inputs = processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)

        t2 = time.time()
        print(f"  Inference on 480p dummy image: {t2-t1:.3f}s")
        print(f"  Depth map shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")

        depth_msg_size = depth.nbytes
        print(f"  Depth image message: {depth_msg_size / 1024:.1f} KB")

        v, u = np.mgrid[0:480:4, 0:640:4]
        z = depth[v, u]
        x = (u - 640/2) * z / 640
        y = (v - 480/2) * z / 480
        mask = (z > 0.1) & np.isfinite(z)
        n_points = mask.sum()
        print(f"  PointCloud (stride 4): {n_points} valid points, "
              f"~{n_points * 12 / 1024:.1f} KB")

        print("  [PASS] Depth model smoke test passed")
        return True

    except Exception as e:
        print(f"  [FAIL] Depth model smoke test failed: {e}")
        return False


def check_ros2():
    print("\n--- ROS2 Environment ---")
    results = []

    ros_distro = os.environ.get("ROS_DISTRO", "")
    results.append(check(f"ROS_DISTRO={ros_distro}", ros_distro == "humble",
                         "Source /opt/ros/jazzy/setup.bash"))

    install_sh = os.path.exists("/workspace/install/setup.bash")
    results.append(check("Workspace install exists", install_sh,
                         "Run colcon build in the workspace"))

    pkg_dirs = {
        "adas_camera": "/workspace/src/adas_camera",
        "adas_depth": "/workspace/src/adas_depth",
        "adas_launch": "/workspace/src/adas_launch",
    }
    for pkg, path in pkg_dirs.items():
        results.append(check(f"Package {pkg} found", os.path.isdir(path)))

    return all(results)


def main():
    print("=" * 60)
    print("ADAS Deployment Diagnostic")
    print("=" * 60)

    checks = [
        ("System", check_system),
        ("Python Dependencies", check_python_deps),
        ("ROS2", check_ros2),
        ("Depth Model", test_dummy_depth),
    ]

    results = []
    for name, fn in checks:
        print(f"\n{'=' * 60}")
        print(f"Section: {name}")
        print(f"{'=' * 60}")
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed. The pipeline should be ready to run.")
        print("Start it with: ros2 launch adas_launch adas_pipeline.launch.py")
    else:
        print("Some checks failed. Fix the issues above before running the pipeline.")
        print("Re-run this diagnostic after making changes.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
