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
        // Suppress INFO level messages
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

    // Step 2: Use the correct input/output tensor names found from Python script
    const char *inputTensorName = "input_0";
    const char *outputTensorName1 = "output_0";
    const char *outputTensorName2 = "1030"; // The second output tensor

    // Step 3: Allocate Device Memory for Input and Outputs
    void *buffers[3];                                                       // Now handling two output buffers
    cudaMalloc(&buffers[0], input_size);                                    // Allocate memory for input
    cudaMemcpy(buffers[0], input_data, input_size, cudaMemcpyHostToDevice); // Copy input data to GPU

    int output_size_1 = 1;                                  // Adjust based on the actual output size of output_0
    cudaMalloc(&buffers[1], output_size_1 * sizeof(float)); // Allocate memory for output_0

    int output_size_2 = 1;                                  // Adjust based on the actual output size of 1030
    cudaMalloc(&buffers[2], output_size_2 * sizeof(float)); // Allocate memory for output_1030

    // Step 4: Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Step 5: Set tensor addresses for both input and outputs
    context->setTensorAddress(inputTensorName, buffers[0]);
    context->setTensorAddress(outputTensorName1, buffers[1]);
    context->setTensorAddress(outputTensorName2, buffers[2]);

    // Step 6: Enqueue inference using enqueueV3
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to enqueue inference using enqueueV3" << std::endl;
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(buffers[2]);
        cudaStreamDestroy(stream);
        return {};
    }

    // Step 7: Synchronize the stream to wait for inference to complete
    cudaStreamSynchronize(stream);

    // Step 8: Copy Results from Device to Host
    std::vector<float> output_data_1(output_size_1);
    cudaMemcpy(output_data_1.data(), buffers[1], output_size_1 * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> output_data_2(output_size_2);
    cudaMemcpy(output_data_2.data(), buffers[2], output_size_2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 9: Parse Output to BoundingBox (for both outputs if needed)
    std::vector<BoundingBox> detected_faces;

    // TODO: Parse output_data_1 and output_data_2 to retrieve bounding boxes
    // Example: detected_faces.push_back({x, y, width, height});

    // Step 10: Clean Up
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream); // Destroy the CUDA stream

    return detected_faces;
}
