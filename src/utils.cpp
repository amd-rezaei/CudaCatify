#include "face_swap.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>

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

// Helper function to calculate total number of elements from tensor dimensions
size_t calculate_num_elements(const nvinfer1::Dims &dims)
{
    size_t num_elements = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        num_elements *= dims.d[i];
    }
    return num_elements;
}

// Get the size of the data type
size_t get_dtype_size(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return 2; // Half-precision (16-bit, 2 bytes)
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    default:
        std::cerr << "Unsupported data type!" << std::endl;
        return 0; // Error case
    }
}

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

    // Step 2: Set the binding indices manually (input is index 0, outputs are 1 and 2)
    int inputIndex = 0;   // Input tensor is at binding index 0
    int outputIndex1 = 1; // First output tensor is at binding index 1
    int outputIndex2 = 2; // Second output tensor is at binding index 2

    // Retrieve the tensor shapes and data types
    nvinfer1::Dims input_dims = engine->getTensorShape("input_0");
    nvinfer1::Dims output_dims1 = engine->getTensorShape("output_0");
    nvinfer1::Dims output_dims2 = engine->getTensorShape("1030");

    nvinfer1::DataType input_dtype = engine->getTensorDataType("input_0");
    nvinfer1::DataType output_dtype1 = engine->getTensorDataType("output_0");
    nvinfer1::DataType output_dtype2 = engine->getTensorDataType("1030");

    // Calculate buffer sizes dynamically
    size_t input_size_bytes = calculate_num_elements(input_dims) * get_dtype_size(input_dtype);
    size_t output_size_bytes1 = calculate_num_elements(output_dims1) * get_dtype_size(output_dtype1);
    size_t output_size_bytes2 = calculate_num_elements(output_dims2) * get_dtype_size(output_dtype2);

    // Check if input size bytes match the input data size
    if (input_size_bytes != input_size)
    {
        std::cerr << "Input size mismatch! Calculated: " << input_size_bytes << " bytes, Provided: " << input_size << " bytes." << std::endl;
        return {};
    }

    // Allocate device memory for input and outputs
    void *buffers[3];
    if (cudaMalloc(&buffers[inputIndex], input_size_bytes) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for input buffer" << std::endl;
        return {};
    }

    if (cudaMalloc(&buffers[outputIndex1], output_size_bytes1) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for output buffer 1" << std::endl;
        return {};
    }

    if (cudaMalloc(&buffers[outputIndex2], output_size_bytes2) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory for output buffer 2" << std::endl;
        return {};
    }

    // Copy input data to GPU
    std::cout << "Copying input data to GPU: " << input_size_bytes << " bytes" << std::endl;
    if (cudaMemcpy(buffers[inputIndex], input_data, input_size_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy input data to GPU!" << std::endl;
        return {};
    }

    // Step 3: Create CUDA stream
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return {};
    }

    // Step 4: Set tensor addresses
    if (!context->setInputTensorAddress("input_0", buffers[inputIndex]))
    {
        std::cerr << "Failed to set input tensor address" << std::endl;
        return {};
    }

    if (!context->setOutputTensorAddress("output_0", buffers[outputIndex1]))
    {
        std::cerr << "Failed to set output tensor address 1" << std::endl;
        return {};
    }

    if (!context->setOutputTensorAddress("1030", buffers[outputIndex2]))
    {
        std::cerr << "Failed to set output tensor address 2" << std::endl;
        return {};
    }

    // Step 5: Enqueue inference using enqueueV3
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to enqueue inference using enqueueV3" << std::endl;
        return {};
    }

    // Synchronize the stream to wait for inference to complete
    cudaStreamSynchronize(stream);

    // Step 6: Copy output data from GPU to host
    std::vector<float> output_data_1(calculate_num_elements(output_dims1));
    if (cudaMemcpy(output_data_1.data(), buffers[outputIndex1], output_size_bytes1, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy output_0 from GPU to host" << std::endl;
        return {};
    }

    std::vector<float> output_data_2(calculate_num_elements(output_dims2));
    if (cudaMemcpy(output_data_2.data(), buffers[outputIndex2], output_size_bytes2, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy output_1030 from GPU to host" << std::endl;
        return {};
    }

    // Parse output to bounding boxes (assume output_data_1 has bounding boxes and output_data_2 has confidence scores)
    std::vector<BoundingBox> detected_faces;
    int num_boxes = output_data_1.size() / 4; // Assuming each box has 4 values: x, y, width, height

    for (int i = 0; i < num_boxes; ++i)
    {
        int idx = i * 4;
        float x = output_data_1[idx];          // x-coordinate
        float y = output_data_1[idx + 1];      // y-coordinate
        float width = output_data_1[idx + 2];  // width
        float height = output_data_1[idx + 3]; // height

        float confidence = output_data_2[i]; // Assuming confidence is in the second output

        if (confidence > 0.5)
        { // Example confidence threshold
            BoundingBox face = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height)};
            detected_faces.push_back(face);
        }
    }

    // Step 7: Clean up
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
    cudaStreamDestroy(stream);

    return detected_faces;
}
