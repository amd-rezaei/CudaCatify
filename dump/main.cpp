#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime header

// Helper function to perform non-maximum suppression (NMS)
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect> &boxes, const std::vector<float> &confidences, float threshold)
{
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, threshold, indices);
    return indices;
}

// Post-process the inference results, draw bounding boxes, and save the image
void postProcessAndSaveImage(cv::Mat &image, const std::vector<float> &outputBoxes, const std::vector<float> &outputScores, const std::vector<int> &outputClasses, int numBoxes, int numClasses, int inputWidth, int inputHeight)
{
    float confThreshold = 0.6f; // Increase the confidence threshold to filter low confidence detections
    float nmsThreshold = 0.5f;  // Adjust NMS threshold to reduce overlapping boxes

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    int imgHeight = image.rows;
    int imgWidth = image.cols;

    // Calculate scaling factors to convert back to original image size
    float xScale = static_cast<float>(imgWidth) / inputWidth;
    float yScale = static_cast<float>(imgHeight) / inputHeight;

    // Calculate padding (if any) that was added during preprocessing
    float scale = std::min(float(inputWidth) / imgWidth, float(inputHeight) / imgHeight);
    int paddingX = (inputWidth - imgWidth * scale) / 2;
    int paddingY = (inputHeight - imgHeight * scale) / 2;

    // Define the class labels
    std::vector<std::string> labels = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
        "bed", "diningtable", "toilet", "TVmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};

    // Iterate over the number of detections
    for (int i = 0; i < numBoxes; ++i)
    {
        // Extract bounding box coordinates
        float x_center = outputBoxes[i * 4];
        float y_center = outputBoxes[i * 4 + 1];
        float width = outputBoxes[i * 4 + 2];
        float height = outputBoxes[i * 4 + 3];

        // Undo the scaling and padding
        int x_min = static_cast<int>((x_center - width / 2) * inputWidth - paddingX) * xScale;
        int y_min = static_cast<int>((y_center - height / 2) * inputHeight - paddingY) * yScale;
        int box_width = static_cast<int>(width * inputWidth * xScale);
        int box_height = static_cast<int>(height * inputHeight * yScale);

        float confidence = outputScores[i]; // Use the actual confidence score from the output
        int classId = outputClasses[i];     // Get the predicted class ID

        // Only consider boxes with confidence greater than the threshold
        if (confidence > confThreshold)
        {
            boxes.emplace_back(x_min, y_min, box_width, box_height);
            confidences.push_back(confidence);
            classIds.push_back(classId);

            // Print out the details of the detected objects with actual class names
            std::cout << "Detected object: " << labels[classId] // Use class name from the labels array
                      << " with confidence: " << confidence
                      << ", Bounding Box [x_min: " << x_min
                      << ", y_min: " << y_min
                      << ", width: " << box_width
                      << ", height: " << box_height
                      << "]" << std::endl;
        }
    }

    // Perform Non-Maximum Suppression (NMS)
    std::vector<int> nmsIndices = nonMaxSuppression(boxes, confidences, nmsThreshold);

    // Draw the boxes on the image
    for (int idx : nmsIndices)
    {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        float confidence = confidences[idx];

        // Draw bounding box
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        // Put label with class name and confidence
        std::string label = labels[classId] + ": " + std::to_string(confidence);
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image, cv::Point(box.x, box.y - labelSize.height),
                      cv::Point(box.x + labelSize.width, box.y + baseline), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Save the result image
    cv::imwrite("output_with_bboxes.jpg", image);
    std::cout << "Saved output image with bounding boxes as 'output_with_bboxes.jpg'" << std::endl;
}

// Preprocess image for ONNX input (resize and normalize)
std::vector<float> preprocessImage(cv::Mat &image, int inputWidth, int inputHeight)
{
    cv::Mat resizedImage;
    int originalHeight = image.rows;
    int originalWidth = image.cols;

    // Resize the image while maintaining the aspect ratio
    float scale = std::min(float(inputWidth) / originalWidth, float(inputHeight) / originalHeight);
    int newWidth = static_cast<int>(originalWidth * scale);
    int newHeight = static_cast<int>(originalHeight * scale);

    // Resize the image and pad it to keep the aspect ratio
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));
    cv::Mat paddedImage = cv::Mat::zeros(inputHeight, inputWidth, CV_32FC3);

    // Center the resized image
    resizedImage.copyTo(paddedImage(cv::Rect((inputWidth - newWidth) / 2, (inputHeight - newHeight) / 2, newWidth, newHeight)));

    // Convert image to float and normalize
    paddedImage.convertTo(paddedImage, CV_32F, 1 / 255.0);

    // Convert HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(paddedImage, channels);

    std::vector<float> inputTensorValues;
    for (int c = 0; c < 3; ++c)
    {
        inputTensorValues.insert(inputTensorValues.end(), (float *)channels[c].datastart, (float *)channels[c].dataend);
    }

    return inputTensorValues;
}

// Run inference using ONNX Runtime
void runInference(const std::string &modelPath, const std::string &imagePath)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv4-Tiny");
    Ort::SessionOptions sessionOptions;
    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output info
    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    const char *inputName = inputNameAllocated.get(); // Extract raw pointer to input name

    auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Input shape: ";
    for (auto s : inputShape)
        std::cout << s << " ";
    std::cout << std::endl;

    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
    const char *outputName = outputNameAllocated.get(); // Extract raw pointer to output name

    auto outputShape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shape: ";
    for (auto s : outputShape)
        std::cout << s << " ";
    std::cout << std::endl;

    int inputWidth = inputShape[3];
    int inputHeight = inputShape[2];

    // Read the input image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return;
    }

    // Preprocess the image
    std::vector<float> inputTensorValues = preprocessImage(image, inputWidth, inputHeight);

    // Prepare input tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size());

    // Run inference
    std::vector<const char *> inputNames = {inputName};   // Wrap inputName in vector
    std::vector<const char *> outputNames = {outputName}; // Wrap outputName in vector
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    if (outputTensors.size() != 1)
    {
        std::cerr << "Expected 1 output tensor, got " << outputTensors.size() << std::endl;
        return;
    }

    // Get outputs (assuming output tensor contains all needed data, including boxes and scores)
    float *outputData = outputTensors[0].GetTensorMutableData<float>();

    // Debug print to verify data structure
    std::cout << "First few output values: ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // Post-process and save the image
    postProcessAndSaveImage(image, std::vector<float>(outputData, outputData + outputShape[1] * 4),
                            std::vector<float>(outputData, outputData + outputShape[1]), // Dummy for now
                            std::vector<int>(outputData, outputData + outputShape[1]),   // Dummy for now
                            outputShape[1], 80, inputWidth, inputHeight);
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
