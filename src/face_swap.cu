#include "face_swap.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nppi.h>
#include <vector>

// TensorRT Logger class definition
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // Only log errors and warnings
        if (severity != Severity::kINFO)
        {
            std::cerr << "TensorRT: " << msg << std::endl;
        }
    }
};

// Assuming you have an engine loaded
void inspect_bindings(nvinfer1::ICudaEngine *engine)
{
    // Loop over the bindings and print their names
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        const char *bindingName = engine->getBindingName(i);
        bool isInput = engine->bindingIsInput(i);
        std::cout << "Binding " << i << ": " << bindingName
                  << (isInput ? " (Input)" : " (Output)") << std::endl;
    }
}

// Load the TensorRT engine from a file
nvinfer1::ICudaEngine *load_engine(const std::string &engine_file, nvinfer1::IRuntime *&runtime)
{
    std::ifstream engine_stream(engine_file, std::ios::binary);
    if (!engine_stream)
    {
        std::cerr << "Error opening engine file: " << engine_file << std::endl;
        return nullptr;
    }

    std::vector<char> engine_data((std::istreambuf_iterator<char>(engine_stream)), std::istreambuf_iterator<char>());

    static Logger gLogger;
    runtime = nvinfer1::createInferRuntime(gLogger);

    if (!runtime)
    {
        std::cerr << "Failed to create runtime" << std::endl;
        return nullptr;
    }

    // Deserialize the engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine)
    {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
    }

    return engine;
}

// Allocate CUDA buffers for input and output
void allocate_buffers(nvinfer1::ICudaEngine *engine, nvinfer1::IExecutionContext *context, std::vector<void *> &buffers)
{
    int nbBindings = engine->getNbBindings(); // Get the number of input/output bindings
    buffers.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = engine->getBindingDimensions(i); // Get dimensions of each binding
        size_t totalSize = 1;
        for (int j = 0; j < dims.nbDims; ++j)
        {
            totalSize *= dims.d[j];
        }

        size_t elementSize = sizeof(float); // Assuming all tensors are float32
        if (cudaMalloc(&buffers[i], totalSize * elementSize) != cudaSuccess)
        {
            std::cerr << "Failed to allocate memory for binding " << i << std::endl;
        }
    }
}

// Run YOLO inference using TensorRT and enqueueV3
std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine *engine, void *input_data, int input_size)
{
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return {};
    }

    // Inspect and print binding names
    inspect_bindings(engine);

    // Allocate input/output buffers
    std::vector<void *> buffers;
    allocate_buffers(engine, context, buffers);

    // Get the binding indices for input and output (assuming you know the names)
    int inputIndex = engine->getBindingIndex("input_0");   // Replace with your actual input tensor name
    int outputIndex = engine->getBindingIndex("output_0"); // Replace with your actual output tensor name

    // Check if the input and output indices are valid
    if (inputIndex == -1 || outputIndex == -1)
    {
        std::cerr << "Failed to find input/output binding indices" << std::endl;
        return {};
    }

    // Create a CUDA stream
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return {};
    }

    // Set the tensor addresses (bind input and output buffers)
    context->setTensorAddress(engine->getBindingName(inputIndex), buffers[inputIndex]);   // Input tensor
    context->setTensorAddress(engine->getBindingName(outputIndex), buffers[outputIndex]); // Output tensor

    // Copy input data to GPU asynchronously
    if (cudaMemcpyAsync(buffers[inputIndex], input_data, input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        std::cerr << "Failed to copy input data to GPU" << std::endl;
        return {};
    }

    // Launch inference using enqueueV3
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to run inference using enqueueV3" << std::endl;
        return {};
    }

    // Retrieve output tensor dimensions
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    int outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        outputSize *= outputDims.d[i];
    }

    // Copy output back from GPU asynchronously
    std::vector<float> output_data(outputSize);
    if (cudaMemcpyAsync(output_data.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        std::cerr << "Failed to copy output data from GPU" << std::endl;
        return {};
    }

    // Synchronize the stream to ensure computation has completed
    cudaStreamSynchronize(stream);

    // Post-process output to extract bounding boxes
    std::vector<BoundingBox> detected_faces;
    for (size_t i = 0; i < output_data.size(); i += 7)
    { // Adjust stride based on YOLO output structure
        BoundingBox bbox;
        bbox.x = static_cast<int>(output_data[i + 3] * 640); // assuming input size is 640x640
        bbox.y = static_cast<int>(output_data[i + 4] * 640);
        bbox.width = static_cast<int>(output_data[i + 5] * 640) - bbox.x;
        bbox.height = static_cast<int>(output_data[i + 6] * 640) - bbox.y;
        detected_faces.push_back(bbox);
    }

    // Free buffers
    for (void *buffer : buffers)
    {
        cudaFree(buffer);
    }

    // Destroy the stream
    cudaStreamDestroy(stream);

    return detected_faces;
}

// Replace detected faces with kitty emoji using NPP
void replace_with_kitty(unsigned char *img, int width, int height, const BoundingBox &face, unsigned char *kitty_emoji, int kitty_width, int kitty_height)
{
    NppiSize size = {face.width, face.height};
    nppiCopy_8u_C3R(kitty_emoji, kitty_width * 3, img + (face.y * width + face.x) * 3, width * 3, size);
}

int main()
{
    // Load the engine
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = load_engine("models/yolov4-tiny.engine", runtime);

    if (!engine)
    {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }

    // Perform inference (Example usage)
    std::vector<BoundingBox> faces = run_inference(engine, nullptr, 0); // Replace with actual input data

    // Free resources
    engine->destroy();
    runtime->destroy();

    return 0;
}
