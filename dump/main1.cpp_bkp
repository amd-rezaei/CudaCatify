#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime header
#include <cstdint>               // For uint16_t
#include <onnxruntime_float16.h> // For MLFloat16

// Helper function to apply sigmoid
float sigmoid(float x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// Function to convert from Float32 to Float16 (no changes)
std::vector<Ort::Float16_t> convertToMLFloat16(const std::vector<float> &input)
{
    std::vector<Ort::Float16_t> float16Data(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        float f32 = input[i];
        uint32_t f32_bits = *(uint32_t *)&f32; // Get float32 bits

        // Perform the conversion from float32 to float16 using IEEE-754
        uint16_t f16_bits = ((f32_bits >> 16) & 0x8000) | ((((f32_bits & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f32_bits >> 13) & 0x03ff);
        float16Data[i] = *reinterpret_cast<Ort::Float16_t *>(&f16_bits); // Assign to MLFloat16
    }
    return float16Data;
}

// Function to convert from Float16 to Float32 (no changes)
std::vector<float> convertFloat16ToFloat32(const Ort::Float16_t *input, size_t size)
{
    std::vector<float> float32Data(size);
    for (size_t i = 0; i < size; ++i)
    {
        uint16_t f16_bits = *(uint16_t *)&input[i];
        uint32_t sign = (f16_bits & 0x8000) << 16;
        uint32_t exponent = (f16_bits & 0x7C00) >> 10;
        uint32_t mantissa = f16_bits & 0x03FF;

        if (exponent == 0)
        {
            // Subnormal number
            float32Data[i] = sign * (mantissa / 1024.0f) * std::pow(2, -14);
        }
        else if (exponent == 0x1F)
        {
            // NaN or infinity
            float32Data[i] = sign | 0x7F800000 | mantissa;
        }
        else
        {
            // Normalized number
            float32Data[i] = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
    }
    return float32Data;
}

// Function to clamp excessively large or invalid values
float clamp(float value, float minValue, float maxValue)
{
    return std::max(minValue, std::min(value, maxValue));
}

// Helper function to print tensor data type in human-readable form
std::string getONNXTensorElementDataType(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "UINT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "INT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "UINT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "INT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "INT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "INT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return "STRING";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return "BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "DOUBLE";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "UINT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "UINT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return "COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return "COMPLEX128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return "BFLOAT16";
    default:
        return "UNKNOWN";
    }
}

// Preprocess image for ONNX input (resize, normalize, convert to CHW format)
std::vector<float> preprocessImage(cv::Mat &image, int inputWidth, int inputHeight)
{
    cv::Mat resizedImage;
    int originalHeight = image.rows;
    int originalWidth = image.cols;

    // Resize the image while maintaining the aspect ratio
    float scale = std::min(float(inputWidth) / originalWidth, float(inputHeight) / originalHeight);
    int newWidth = static_cast<int>(originalWidth * scale);
    int newHeight = static_cast<int>(originalHeight * scale);

    // Resize the image and pad it to keep the aspect ratio (use 128 as padding value)
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));
    cv::Mat paddedImage(inputHeight, inputWidth, CV_32FC3, cv::Scalar(128, 128, 128));

    // Center the resized image
    resizedImage.copyTo(paddedImage(cv::Rect((inputWidth - newWidth) / 2, (inputHeight - newHeight) / 2, newWidth, newHeight)));

    // Convert image to float and normalize
    paddedImage.convertTo(paddedImage, CV_32F, 1 / 255.0);

    // Convert HWC (height, width, channels) to CHW (channels, height, width) format
    std::vector<cv::Mat> channels(3);
    cv::split(paddedImage, channels);

    std::vector<float> inputTensorValues;
    for (int c = 0; c < 3; ++c)
    {
        inputTensorValues.insert(inputTensorValues.end(), (float *)channels[c].datastart, (float *)channels[c].dataend);
    }

    return inputTensorValues; // Tensor values in CHW format
}

// Post-process and save the image with bounding boxes (updated logic)
void postProcessAndSaveImage(cv::Mat &image, const std::vector<float> &output, int numBoxes, int numClasses, int imgWidth, int imgHeight)
{
    float confThreshold = 0.5f; // Confidence threshold
    float nmsThreshold = 0.4f;  // NMS threshold

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    int imageHeight = image.rows;
    int imageWidth = image.cols;

    std::cout << "Image dimensions: " << imageWidth << "x" << imageHeight << std::endl;
    std::cout << "Number of boxes: " << numBoxes << ", Number of classes: " << numClasses << std::endl;

    // Iterate over the predictions
    for (int i = 0; i < numBoxes; ++i)
    {
        int index = i * 6; // Each prediction has exactly 6 values

        // Check if the index is within bounds of the output vector
        if (index + 5 >= output.size())
        {
            std::cerr << "Invalid access at index " << index << ", skipping..." << std::endl;
            continue;
        }

        // Objectness score (apply sigmoid here)
        float confidence = sigmoid(output[index + 4]);
        std::cout << "Confidence for box " << i << ": " << confidence << std::endl;

        // Only consider boxes with objectness score greater than the threshold
        if (confidence >= confThreshold)
        {
            // Extract bounding box coordinates (center_x, center_y, width, height) without sigmoid
            float x_center = clamp(output[index] * imgWidth, 0, imgWidth);
            float y_center = clamp(output[index + 1] * imgHeight, 0, imgHeight);

            // Apply exp() to width and height, but first clamp the raw values to avoid overflow
            float raw_width = clamp(output[index + 2], -10.0f, 10.0f);
            float raw_height = clamp(output[index + 3], -10.0f, 10.0f);
            float width = std::exp(raw_width) * imgWidth;
            float height = std::exp(raw_height) * imgHeight;

            // Validate bounding box values
            if (width > 0 && height > 0 && x_center >= 0 && y_center >= 0)
            {
                int x_min = static_cast<int>(x_center - width / 2.0);
                int y_min = static_cast<int>(y_center - height / 2.0);
                boxes.emplace_back(x_min, y_min, static_cast<int>(width), static_cast<int>(height));
                confidences.push_back(confidence);
                classIds.push_back(0); // Assuming single class
            }
        }
    }

    // Perform Non-Maximum Suppression (NMS)
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

    // Draw bounding boxes
    for (int idx : nmsIndices)
    {
        cv::Rect box = boxes[idx];
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Object: " + std::to_string(confidences[idx]);
        cv::putText(image, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Save output image
    cv::imwrite("output_with_bboxes.jpg", image);
    std::cout << "Saved output image with bounding boxes as 'output_with_bboxes.jpg'" << std::endl;
}



void runInference(const std::string &modelPath, const std::string &imagePath)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    Ort::SessionOptions sessionOptions;
    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output info
    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    const char *inputName = inputNameAllocated.get();
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
    const char *outputName = outputNameAllocated.get();

    // Get input shape
    auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Input shape: ";
    for (auto s : inputShape)
        std::cout << s << " ";
    std::cout << std::endl;

    // Read input image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return;
    }

    // Preprocess the image
    std::vector<float> inputTensorValues = preprocessImage(image, inputShape[3], inputShape[2]);

    // Convert float32 to MLFloat16
    std::vector<Ort::Float16_t> inputTensorValuesFloat16 = convertToMLFloat16(inputTensorValues);

    // Create input tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memoryInfo, inputTensorValuesFloat16.data(), inputTensorValuesFloat16.size(),
        inputShape.data(), inputShape.size());

    // Run inference
    std::vector<const char *> inputNames = {inputName};
    std::vector<const char *> outputNames = {outputName};

    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    if (outputTensors.size() != 1)
    {
        std::cerr << "Expected 1 output tensor, got " << outputTensors.size() << std::endl;
        return;
    }

    // Get tensor type and shape
    auto outputTypeInfo = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputDataType = outputTypeInfo.GetElementType();
    std::cout << "Output data type: " << getONNXTensorElementDataType(outputDataType) << std::endl;

    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shape: ";
    for (auto s : outputShape)
        std::cout << s << " ";
    std::cout << std::endl;

    // Extract output as Float16
    auto *outputDataFloat16 = outputTensors[0].GetTensorMutableData<Ort::Float16_t>();

    // Convert float16 output to float32 for post-processing
    size_t outputSize = outputShape[1] * outputShape[2];
    std::vector<float> outputData = convertFloat16ToFloat32(outputDataFloat16, outputSize);

    // Post-process and save the image
    postProcessAndSaveImage(image, outputData, outputShape[1], 80, inputShape[3], inputShape[2]);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <image>" << std::endl;
        return -1;
    }

    runInference(argv[1], argv[2]);

    return 0;
}
 