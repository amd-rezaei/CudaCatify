import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        print(f"Loading ONNX file from path {onnx_file_path}...")
        if not parser.parse(model.read()):
            print('Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("Building TensorRT engine...")
    builder.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_cuda_engine(network)

    if engine:
        print(f"Saving TensorRT engine to {engine_file_path}...")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("TensorRT engine saved.")
    else:
        print("Failed to build the TensorRT engine.")
    
    return engine

if __name__ == "__main__":
    onnx_model_path = "../models/yolov4-tiny.onnx"  # Update this path if necessary
    engine_file_path = "../models/yolov4-tiny.engine"
    
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}. Please make sure the file exists.")
    else:
        build_engine(onnx_model_path, engine_file_path)
