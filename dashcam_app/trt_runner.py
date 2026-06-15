import numpy as np
import os
import time

TRT_AVAILABLE = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    pass


class TrtRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = None
        self.output_shapes = []
        self.d_input = None
        self.d_outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                self.input_shape = shape
            else:
                self.output_shapes.append(shape)

        print(f"[TRT] Engine loaded: input={self.input_shape}, outputs={self.output_shapes}", flush=True)

    def infer(self, input_np):
        import torch
        if self.d_input is None:
            self.d_input = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
            self.d_outputs = [torch.empty(s, dtype=torch.float32, device='cuda') for s in self.output_shapes]
            self.bindings = [self.d_input.data_ptr()] + [o.data_ptr() for o in self.d_outputs]
        t = torch.from_numpy(np.ascontiguousarray(input_np))
        self.d_input.copy_(t)
        self.context.execute_v2(self.bindings)
        return self.d_outputs
