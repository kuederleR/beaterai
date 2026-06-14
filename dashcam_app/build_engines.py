"""
Build TensorRT engines for models (YOLOPv2 ONNX → TRT).

Inside container:
    docker compose run --rm dashcam python3 build_engines.py
"""

import os
import sys

os.makedirs("models", exist_ok=True)


def build_engine_from_onnx(onnx_path, engine_path, fp16=True):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] Parsing {onnx_path} ...", flush=True)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"[build] ONNX parse error: {parser.get_error(err)}", flush=True)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled", flush=True)
    else:
        print("[build] FP16 not available, using FP32", flush=True)

    print("[build] Building engine (this takes a few minutes)...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[build] Engine build failed", flush=True)
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build] Engine saved to {engine_path}", flush=True)


if __name__ == "__main__":
    print("[build] YOLOPv2 uses its own PyTorch JIT → TRT build at runtime.")
    print("[build] To manually convert an ONNX to TRT, use:")
    print("[build]   python3 -c \"from build_engines import build_engine_from_onnx; "
          "build_engine_from_onnx('model.onnx', 'model.engine')\"")
