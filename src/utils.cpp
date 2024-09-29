#include "face_swap.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory> // For std::unique_ptr

// Implement the logger
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO)
        {
            std::cerr << "TensorRT: " << msg << std::endl;
        }
    }
};

// Global logger instance
Logger gLogger;

// Load TensorRT engine from file
nvinfer1::ICudaEngine *load_engine(const std::string &engine_file, nvinfer1::IRuntime *&runtime)
{
    std::ifstream engineFile(engine_file, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Failed to open engine file: " << engine_file << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engine_data(fsize);
    engineFile.read(engine_data.data(), fsize);

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
    {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return nullptr;
    }

    std::cout << "TensorRT engine loaded successfully!" << std::endl;

    return runtime->deserializeCudaEngine(engine_data.data(), fsize); // Deserialize engine
}

// Run inference on the loaded engine using enqueueV3
std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine *engine, void *input_data, int input_size)
{
    // Step 1: Create Execution Context
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return {};
    }
    std::cout << "Execution context created successfully!" << std::endl;

    // Step 2: Set the binding indices manually (input is index 0, outputs are 1 and 2)
    int inputIndex = 0;   // Input tensor is at binding index 0
    int outputIndex1 = 1; // First output tensor is at binding index 1
    int outputIndex2 = 2; // Second output tensor is at binding index 2

    std::cout << "Input Index: " << inputIndex << ", Output Index 1: " << outputIndex1 << ", Output Index 2: " << outputIndex2 << std::endl;

    // Step 3: Allocate device memory for input and outputs
    void *buffers[3];
    if (cudaMalloc(&buffers[inputIndex], input_size) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for input buffer" << std::endl;
        return {};
    }
    std::cout << "Input buffer allocated successfully!" << std::endl;

    // Copy input data to GPU
    if (cudaMemcpy(buffers[inputIndex], input_data, input_size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy input data to GPU" << std::endl;
        return {};
    }
    std::cout << "Input data copied to GPU successfully!" << std::endl;

    // Assuming reasonable output sizes; adjust based on your model
    int output_size_1 = 1000; // Adjust based on the actual output size
    int output_size_2 = 1000; // Adjust based on the actual output size

    if (cudaMalloc(&buffers[outputIndex1], output_size_1 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for output buffer 1" << std::endl;
        return {};
    }
    std::cout << "Output buffer 1 allocated successfully!" << std::endl;

    if (cudaMalloc(&buffers[outputIndex2], output_size_2 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for output buffer 2" << std::endl;
        return {};
    }
    std::cout << "Output buffer 2 allocated successfully!" << std::endl;

    // Step 4: Create CUDA stream
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return {};
    }
    std::cout << "CUDA stream created successfully!" << std::endl;

    // Step 5: Use the `bindings[]` array to reference input and output buffers
    void *bindings[] = {buffers[inputIndex], buffers[outputIndex1], buffers[outputIndex2]};

    // Step 6: Enqueue inference using `enqueueV3()`
    std::cout << "Attempting to enqueue inference using enqueueV3..." << std::endl;
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to enqueue inference using enqueueV3" << std::endl;
        cudaFree(buffers[inputIndex]);
        cudaFree(buffers[outputIndex1]);
        cudaFree(buffers[outputIndex2]);
        cudaStreamDestroy(stream);
        return {};
    }
    std::cout << "Inference enqueued successfully!" << std::endl;

    // Step 7: Synchronize the stream to wait for inference to complete
    cudaStreamSynchronize(stream);
    std::cout << "CUDA stream synchronized!" << std::endl;

    // Step 8: Copy output data from GPU to host
    std::vector<float> output_data_1(output_size_1);
    if (cudaMemcpy(output_data_1.data(), buffers[outputIndex1], output_size_1 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy output_0 from GPU to host" << std::endl;
        return {};
    }
    std::cout << "Output data 1 copied from GPU to host!" << std::endl;

    std::vector<float> output_data_2(output_size_2);
    if (cudaMemcpy(output_data_2.data(), buffers[outputIndex2], output_size_2 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy output_1030 from GPU to host" << std::endl;
        return {};
    }
    std::cout << "Output data 2 copied from GPU to host!" << std::endl;

    // Step 9: Parse output to bounding boxes (for both outputs if needed)
    std::vector<BoundingBox> detected_faces;
    int num_boxes = output_size_1 / 4; // Assuming each box has 4 values: x, y, width, height

    std::cout << "Number of boxes detected: " << num_boxes << std::endl;

    for (int i = 0; i < num_boxes; ++i)
    {
        int idx = i * 4;
        float x = output_data_1[idx];          // x-coordinate
        float y = output_data_1[idx + 1];      // y-coordinate
        float width = output_data_1[idx + 2];  // width
        float height = output_data_1[idx + 3]; // height

        float confidence = output_data_2[i]; // Assuming confidence is in the second output

        std::cout << "Box " << i << ": (x: " << x << ", y: " << y << ", w: " << width << ", h: " << height
                  << ", confidence: " << confidence << ")" << std::endl;

        if (confidence > 0.5) // Example confidence threshold
        {
            BoundingBox face = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height)};
            detected_faces.push_back(face);
        }
    }

    // Step 10: Clean up
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
    cudaStreamDestroy(stream); // Destroy the CUDA stream

    return detected_faces;
}
