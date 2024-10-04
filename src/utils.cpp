#include "face_swap.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <map>   // For storing class labels
#include <cmath> // For std::isnan and std::isinf

// Define constants for YOLO input size
const int YOLO_INPUT_WIDTH = 416;
const int YOLO_INPUT_HEIGHT = 416;

// Function to validate input data
bool isInputDataValid(const void *input_data_void, size_t input_size)
{
    // Print the input data pointer and size
    std::cout << "Validating input data pointer: " << input_data_void << std::endl;
    std::cout << "Input data size in bytes: " << input_size << std::endl;

    if (input_data_void == nullptr)
    {
        std::cerr << "Error: Input data pointer is null!" << std::endl;
        return false;
    }

    size_t num_elements = input_size / sizeof(float); // Assuming float elements

    // Debugging: Print the first few bytes as raw data
    const unsigned char *raw_data = static_cast<const unsigned char *>(input_data_void);
    std::cout << "First few raw input bytes: ";
    for (size_t i = 0; i < std::min(input_size, size_t(10)); ++i)
    {
        std::cout << std::hex << static_cast<int>(raw_data[i]) << " ";
    }
    std::cout << std::dec << std::endl; // Reset to decimal

    if (input_size == 0)
    {
        std::cerr << "Error: Input data size is zero!" << std::endl;
        return false;
    }

    return true;
}

// Function to check if bounding boxes and confidence scores are valid
bool isOutputDataValid(const std::vector<float> &output_data)
{
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        if (std::isnan(output_data[i]) || std::isinf(output_data[i]))
        {
            std::cerr << "Error: Invalid value detected in output data at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

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

// Define class labels (for example, for YOLO this can be based on COCO dataset)
std::map<int, std::string> class_labels = {
    {0, "Person"},
    {1, "Bicycle"},
    {2, "Car"},
    {3, "Motorbike"},
    {4, "Airplane"},
    {5, "Bus"},
    // ... Add more labels as per your model
};

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
    // Open the engine file in binary mode
    std::ifstream engineFile(engine_file, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Failed to open engine file: " << engine_file << std::endl;
        return nullptr;
    }

    // Get the size of the engine file
    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    // Log the file size for debugging
    std::cout << "Engine file size: " << fsize << " bytes" << std::endl;

    // Check if the file size is valid
    if (fsize <= 0)
    {
        std::cerr << "Invalid engine file size: " << fsize << std::endl;
        return nullptr;
    }

    // Read the file contents into a buffer
    std::vector<char> engine_data(fsize);
    engineFile.read(engine_data.data(), fsize);

    // Check if the file was read successfully
    if (!engineFile)
    {
        std::cerr << "Failed to read engine file: " << engine_file << std::endl;
        return nullptr;
    }

    // Create the TensorRT runtime
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
    {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return nullptr;
    }

    // Deserialize the engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), fsize);
    if (!engine)
    {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return nullptr;
    }

    return engine;
}

// Function to check for CUDA errors
void checkCudaError(const char *message)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << message << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine *engine, void *input_data, int input_size)
{
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return {};
    }

    int inputIndex = 0;
    int outputIndex1 = 1; // For bounding boxes
    int outputIndex2 = 2; // For confidence scores

    nvinfer1::Dims input_dims = engine->getTensorShape("input_0");
    nvinfer1::Dims output_dims1 = engine->getTensorShape("output_0"); // Bounding boxes
    nvinfer1::Dims output_dims2 = engine->getTensorShape("1030");     // Confidence scores

    size_t input_size_bytes = calculate_num_elements(input_dims) * get_dtype_size(engine->getTensorDataType("input_0"));
    size_t output_size_bytes1 = calculate_num_elements(output_dims1) * get_dtype_size(engine->getTensorDataType("output_0"));
    size_t output_size_bytes2 = calculate_num_elements(output_dims2) * get_dtype_size(engine->getTensorDataType("1030"));

    std::cout << "Input tensor size: " << input_size_bytes << " bytes" << std::endl;
    std::cout << "Output bounding boxes size: " << output_size_bytes1 << " bytes" << std::endl;
    std::cout << "Output confidence scores size: " << output_size_bytes2 << " bytes" << std::endl;

    if (input_size_bytes != input_size)
    {
        std::cerr << "Input size mismatch!" << std::endl;
        return {};
    }

    void *buffers[3];

    // Allocate memory for input and output buffers
    if (cudaMalloc(&buffers[inputIndex], input_size_bytes) != cudaSuccess)
    {
        std::cerr << "Failed to allocate CUDA memory for input" << std::endl;
        return {};
    }
    checkCudaError("After cudaMalloc for input");

    if (cudaMalloc(&buffers[outputIndex1], output_size_bytes1) != cudaSuccess)
    {
        std::cerr << "Failed to allocate CUDA memory for output bounding boxes" << std::endl;
        cudaFree(buffers[inputIndex]);
        return {};
    }
    checkCudaError("After cudaMalloc for output bounding boxes");

    if (cudaMalloc(&buffers[outputIndex2], output_size_bytes2) != cudaSuccess)
    {
        std::cerr << "Failed to allocate CUDA memory for output confidence scores" << std::endl;
        cudaFree(buffers[inputIndex]);
        cudaFree(buffers[outputIndex1]);
        return {};
    }
    checkCudaError("After cudaMalloc for output confidence scores");

    std::cout << "Allocated memory for input and output tensors\n";

    // Validate input data
    std::cout << "Checking input data for validity\n";
    if (!isInputDataValid(input_data, input_size_bytes))
    {
        std::cerr << "Invalid input data detected." << std::endl;
        return {};
    }

    std::cout << "Input data is valid\n";

    // Copy input data to the device
    cudaMemcpy(buffers[inputIndex], input_data, input_size_bytes, cudaMemcpyHostToDevice);
    checkCudaError("After cudaMemcpy for input data");
    std::cout << "Input data copied to GPU\n";

    // Create and configure CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setInputTensorAddress("input_0", buffers[inputIndex]);
    context->setOutputTensorAddress("output_0", buffers[outputIndex1]);
    context->setOutputTensorAddress("1030", buffers[outputIndex2]);

    // Run inference
    if (!context->enqueueV3(stream))
    {
        std::cerr << "Failed to enqueue inference." << std::endl;
        return {};
    }
    cudaStreamSynchronize(stream);
    checkCudaError("After inference");

    std::cout << "Inference completed, copying output data\n";

    // Allocate host memory for output data
    std::vector<float> output_data_1(calculate_num_elements(output_dims1));
    std::vector<float> output_data_2(calculate_num_elements(output_dims2));

    // Copy output data from the device
    cudaMemcpy(output_data_1.data(), buffers[outputIndex1], output_size_bytes1, cudaMemcpyDeviceToHost);
    checkCudaError("After cudaMemcpy for output bounding boxes");
    cudaMemcpy(output_data_2.data(), buffers[outputIndex2], output_size_bytes2, cudaMemcpyDeviceToHost);
    checkCudaError("After cudaMemcpy for output confidence scores");

    // Validate output data
    if (!isOutputDataValid(output_data_1) || !isOutputDataValid(output_data_2))
    {
        std::cerr << "Invalid output data detected!" << std::endl;
        return {};
    }

    std::vector<BoundingBox> detected_faces;
    int num_boxes = output_data_1.size() / 4;

    // Process bounding boxes
    for (int i = 0; i < num_boxes; ++i)
    {
        int idx = i * 4;
        if (idx >= output_data_1.size())
            break;

        float x = output_data_1[idx];
        float y = output_data_1[idx + 1];
        float width = output_data_1[idx + 2];
        float height = output_data_1[idx + 3];

        if (x < 0 || y < 0 || width <= 0 || height <= 0 || (x + width) > YOLO_INPUT_WIDTH || (y + height) > YOLO_INPUT_HEIGHT)
        {
            std::cerr << "Skipping invalid bounding box: (" << x << ", " << y << ", " << width << ", " << height << ")" << std::endl;
            continue;
        }

        float max_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < 80; ++c)
        {
            int score_index = i * 80 + c;
            if (score_index >= output_data_2.size())
                break;

            float score = output_data_2[score_index];
            if (score > max_score)
            {
                max_score = score;
                best_class = c;
            }
        }

        if (best_class == 0 && max_score > 0.5) // Assuming class 0 is 'person'
        {
            detected_faces.push_back({static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height)});
        }
    }

    // Free allocated memory
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
    cudaStreamDestroy(stream);

    return detected_faces;
}
