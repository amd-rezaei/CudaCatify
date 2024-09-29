import tensorrt as trt

# Load the engine
with open('../models/yolov4-tiny.engine', 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Get the number of I/O tensors
num_tensors = engine.num_io_tensors

# Function to calculate tensor size (number of elements)
def calculate_tensor_size(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size

# Iterate over tensors and print their names, modes, shapes, and sizes
for i in range(num_tensors):
    tensor_name = engine.get_tensor_name(i)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    tensor_shape = engine.get_tensor_shape(tensor_name)  # Get tensor shape
    tensor_dtype = engine.get_tensor_dtype(tensor_name)  # Get tensor data type
    bytes_per_component = engine.get_tensor_bytes_per_component(tensor_name)  # Bytes per component (per element)

    tensor_size = calculate_tensor_size(tensor_shape)  # Calculate number of elements
    total_memory_size = tensor_size * bytes_per_component  # Calculate total memory size in bytes

    # Determine if the tensor is input or output
    if tensor_mode == trt.TensorIOMode.INPUT:
        print(f"Input Tensor: {tensor_name}")
    else:
        print(f"Output Tensor: {tensor_name}")

    # Print tensor information
    print(f"  Shape: {tensor_shape}")
    print(f"  Data Type: {tensor_dtype}")
    print(f"  Bytes per Component: {bytes_per_component}")
    print(f"  Total Elements: {tensor_size}")
    print(f"  Memory Size (Bytes): {total_memory_size}")
