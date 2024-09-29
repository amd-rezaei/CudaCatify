import tensorrt as trt

# Load the engine
with open('../models/yolov4-tiny.engine', 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Number of I/O tensors
num_tensors = engine.num_io_tensors

# Iterate over all tensors and print whether they are input or output
for i in range(num_tensors):
    tensor_name = engine.get_tensor_name(i)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    
    if tensor_mode == trt.TensorIOMode.INPUT:
        print(f"Input Tensor: {tensor_name}")
    else:
        print(f"Output Tensor: {tensor_name}")
