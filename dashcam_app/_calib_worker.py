"""
Worker subprocess for INT8 calibration cache generation.
Launched by build_engines._build_calibration_cache — can be killed
early once the cache file is written, saving ~30 min of engine optimization.
"""

import os
import sys
import numpy as np

import tensorrt as trt
import torch


class _FrameCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, npy_path, batch_size=1, cache_path=None):
        super().__init__()
        self._batch_size = batch_size
        self._cache_path = cache_path
        self._idx = 0

        data = np.load(npy_path)
        self._total = len(data)
        n, c, h, w = data.shape
        self._gpu_buf = torch.empty(n, c, h, w, dtype=torch.float32, device='cuda')
        self._gpu_buf.copy_(torch.from_numpy(data).cuda())

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names):
        if self._idx >= self._total:
            return None
        batch_end = min(self._idx + self._batch_size, self._total)
        elements_per_sample = int(np.prod(self._gpu_buf.shape[1:]))
        offset_bytes = self._idx * elements_per_sample * 4
        ptr = int(self._gpu_buf.data_ptr()) + offset_bytes
        self._idx = batch_end
        print(f"[calib] batch {self._idx}/{self._total}", flush=True)
        return [ptr]

    def read_calibration_cache(self):
        if self._cache_path and os.path.exists(self._cache_path):
            with open(self._cache_path, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self._cache_path:
            os.makedirs(os.path.dirname(self._cache_path) or '.', exist_ok=True)
            with open(self._cache_path, 'wb') as f:
                f.write(cache)
            print(f"[build] Calibration cache saved to {self._cache_path}", flush=True)


def main():
    onnx_path, calib_npy_path, cache_path = sys.argv[1:4]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"[build] ONNX parse error: {parser.get_error(err)}", flush=True)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = _FrameCalibrator(calib_npy_path, batch_size=1, cache_path=cache_path)
    config.int8_calibrator = calibrator

    print("[build] Running calibration (collecting activation statistics) ...",
          flush=True)
    builder.build_serialized_network(network, config)


if __name__ == "__main__":
    main()
