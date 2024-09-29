#include "face_swap.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nppi.h>
#include <vector>

// TensorRT Logger
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        std::cerr << "TensorRT: " << msg << std::endl;
    }
}

// Load the TensorRT engine from a file
nvinfer1::ICudaEngine* load_engine(const std::string& engine_file) {
    std::ifstream engine_stream(engine_file, std::ios::binary);
    if (!engine_stream) {
        std::cerr << "Error opening engine file: " << engine_file << std::endl;
        return nullptr;
    }

    std::vector<char> engine_data((std::istreambuf_iterator<char>(engine_stream)), std::istreambuf_iterator<char>());

    // Pass the Logger as a reference
    static Logger gLogger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        return nullptr;
    }

    // Deserialize the engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
    }

    return engine;
}

// Allocate CUDA buffers for input and output
void allocate_buffers(nvinfer1::ICudaEngine* engine, void** buffers, int input_size, int output_size) {
    cudaMalloc(&buffers[0], input_size);  // Input buffer
    cudaMalloc(&buffers[1], output_size); // Output buffer
}

// Run YOLO inference using TensorRT and enqueueV3
std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine* engine, void* input_data, int input_size) {
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return {};
    }

    void* buffers[2];
    int output_size = 1000 * sizeof(float);  // Example size, adjust according to YOLO output
    allocate_buffers(engine, buffers, input_size, output_size);

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Bind input and output buffers by tensor names
    context->setTensorAddress("input_0", buffers[0]);
    context->setTensorAddress("output_0", buffers[1]);

    // Copy input data to GPU asynchronously
    cudaMemcpyAsync(buffers[0], input_data, input_size, cudaMemcpyHostToDevice, stream);

    // Launch inference using enqueueV3
    context->enqueueV3(stream);

    // Copy output back from GPU asynchronously
    std::vector<float> output_data(output_size / sizeof(float));
    cudaMemcpyAsync(output_data.data(), buffers[1], output_size, cudaMemcpyDeviceToHost, stream);

    // Synchronize the stream to ensure computation has completed
    cudaStreamSynchronize(stream);

    // Post-process output to extract bounding boxes
    std::vector<BoundingBox> detected_faces;
    for (size_t i = 0; i < output_data.size(); i += 7) {
        BoundingBox bbox;
        bbox.x = static_cast<int>(output_data[i + 3] * 640);
        bbox.y = static_cast<int>(output_data[i + 4] * 640);
        bbox.width = static_cast<int>(output_data[i + 5] * 640) - bbox.x;
        bbox.height = static_cast<int>(output_data[i + 6] * 640) - bbox.y;
        detected_faces.push_back(bbox);
    }

    // Free buffers
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // Destroy the stream
    cudaStreamDestroy(stream);

    return detected_faces;
}

// Replace detected faces with kitty emoji using NPP
void replace_with_kitty(unsigned char* img, int width, int height, const BoundingBox& face, unsigned char* kitty_emoji, int kitty_width, int kitty_height) {
    NppiSize size = {face.width, face.height};
    nppiCopy_8u_C3R(kitty_emoji, kitty_width * 3, img + (face.y * width + face.x) * 3, width * 3, size);
}
