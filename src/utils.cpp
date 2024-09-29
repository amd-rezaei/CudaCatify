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

// Run inference on the loaded engine
std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine *engine, void *input_data, int input_size)
{
    // Step 1: Create Execution Context
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return {};
    }
    std::cout << "Execution context created successfully!" << std::endl;

    // Step 2: Check binding names for input/output tensors
    std::cout << "Input Tensor: " << engine->getIOTensorName(0) << std::endl;
    std::cout << "Output Tensor 1: " << engine->getIOTensorName(1) << std::endl;
    std::cout << "Output Tensor 2: " << engine->getIOTensorName(2) << std::endl;

    const char *inputTensorName = "input_0";
    const char *outputTensorName1 = "output_0";
    const char *outputTensorName2 = "1030"; // The second output tensor

    // Step 3: Allocate Device Memory for Input and Outputs
    void *buffers[3];
    if (cudaMalloc(&buffers[0], input_size) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for input buffer" << std::endl;
        return {};
    }
    std::cout << "Input buffer allocated successfully!" << std::endl;

    if (cudaMemcpy(buffers[0], input_data, input_size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy input data to GPU" << std::endl;
        return {};
    }
    std::cout << "Input data copied to GPU successfully!" << std::endl;

    int output_size_1 = 1; // Adjust this based on the actual output size of output_0
    int output_size_2 = 1; // Adjust this based on the actual output size of 1030

    if (cudaMalloc(&buffers[1], output_size_1 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for output buffer 1" << std::endl;
        return {};
    }
    std::cout << "Output buffer 1 allocated successfully!" << std::endl;

    if (cudaMalloc(&buffers[2], output_size_2 * sizeof(float)) != cudaSuccess)
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

    // Step 5: Set tensor addresses for both input and outputs
    if (!context->setTensorAddress(inputTensorName, buffers[0]))
    {
        std::cerr << "Failed to set tensor address for input" << std::endl;
        return {};
    }
    if (!context->setTensorAddress(outputTensorName1, buffers[1]))
    {
        std::cerr << "Failed to set tensor address for output_0" << std::endl;
        return {};
    }
    if (!context->setTensorAddress(outputTensorName2, buffers[2]))
    {
        std::cerr << "Failed to set tensor address for output_1030" << std::endl;
        return {};
    }
    std::cout << "Tensor addresses set successfully!" << std::endl;

    // Step 6: Enqueue inference using enqueueV3
    std::cout << "Attempting to enqueue inference..." << std::endl;
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to enqueue inference using enqueueV3" << std::endl;
        std::cerr << "TensorRT Error: Error Code 1: Cuda Runtime (invalid argument)" << std::endl;
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(buffers[2]);
        cudaStreamDestroy(stream);
        return {};
    }
    std::cout << "Inference enqueued successfully!" << std::endl;

    // Step 7: Synchronize the stream to wait for inference to complete
    cudaStreamSynchronize(stream);
    std::cout << "CUDA stream synchronized!" << std::endl;

    // Step 8: Copy Results from Device to Host
    std::vector<float> output_data_1(output_size_1);
    if (cudaMemcpy(output_data_1.data(), buffers[1], output_size_1 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy output_0 from GPU to host" << std::endl;
        return {};
    }
    std::cout << "Output data 1 copied from GPU to host!" << std::endl;

    std::vector<float> output_data_2(output_size_2);
    if (cudaMemcpy(output_data_2.data(), buffers[2], output_size_2 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy output_1030 from GPU to host" << std::endl;
        return {};
    }
    std::cout << "Output data 2 copied from GPU to host!" << std::endl;

    // Step 9: Parse Output to BoundingBox (for both outputs if needed)
    std::vector<BoundingBox> detected_faces;

    int num_boxes = output_size_1 / 4; // Assuming each box has 4 values: (x, y, width, height)
    std::cout << "Number of boxes detected: " << num_boxes << std::endl;

    // Loop over the boxes
    for (int i = 0; i < num_boxes; ++i)
    {
        int idx = i * 4;
        float x = output_data_1[idx];          // x-coordinate
        float y = output_data_1[idx + 1];      // y-coordinate
        float width = output_data_1[idx + 2];  // width
        float height = output_data_1[idx + 3]; // height

        // Retrieve confidence from output_data_2
        float confidence = output_data_2[i];

        std::cout << "Box " << i << ": (x: " << x << ", y: " << y << ", w: " << width << ", h: " << height
                  << ", confidence: " << confidence << ")" << std::endl;

        // Only consider bounding boxes with a high confidence score (e.g., > 0.5)
        if (confidence > 0.5)
        {
            BoundingBox face = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height)};
            detected_faces.push_back(face);
        }
    }

    // Step 10: Clean Up
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream); // Destroy the CUDA stream

    return detected_faces;
}
