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

    // Step 2: Get binding names dynamically
    const char *inputTensorName = nullptr;
    const char *outputTensorName = nullptr;

    // Iterate over all tensors and find input/output names
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char *tensorName = engine->getIOTensorName(i);
        if (engine->getTensorMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputTensorName = tensorName;
        }
        else
        {
            outputTensorName = tensorName;
        }
    }

    if (!inputTensorName || !outputTensorName)
    {
        std::cerr << "Failed to get input/output tensor names from engine" << std::endl;
        return {};
    }

    // Step 3: Allocate Device Memory for Input and Output
    void *buffers[2];
    cudaMalloc(&buffers[0], input_size);                                    // Allocate memory for input
    cudaMemcpy(buffers[0], input_data, input_size, cudaMemcpyHostToDevice); // Copy input data to GPU

    int output_size = 1;                                  // Adjust the output size based on the actual model's output size
    cudaMalloc(&buffers[1], output_size * sizeof(float)); // Allocate memory for output

    // Step 4: Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Step 5: Set tensor addresses using tensor names
    context->setTensorAddress(inputTensorName, buffers[0]);
    context->setTensorAddress(outputTensorName, buffers[1]);

    // Step 6: Enqueue inference using enqueueV3
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to enqueue inference using enqueueV3" << std::endl;
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaStreamDestroy(stream);
        return {};
    }

    // Step 7: Synchronize the stream to wait for inference to complete
    cudaStreamSynchronize(stream);

    // Step 8: Copy Results from Device to Host
    std::vector<float> output_data(output_size);
    cudaMemcpy(output_data.data(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 9: Parse Output to BoundingBox
    std::vector<BoundingBox> detected_faces;

    // TODO: Parse the output_data to retrieve bounding boxes
    // Example: detected_faces.push_back({x, y, width, height});

    // Step 10: Clean Up
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaStreamDestroy(stream); // Destroy the CUDA stream

    return detected_faces;
}
