#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime header

// Helper function to perform non-maximum suppression (NMS)
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect> &boxes, const std::vector<float> &confidences, float confThreshold, float nmsThreshold)
{
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    return indices;
}

// Helper function to apply sigmoid
float sigmoid(float x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

void postProcessAndSaveImage(cv::Mat &image, const std::vector<float> &outputBoxes, const std::vector<float> &outputScores, const std::vector<int> &outputClasses, int numBoxes, int numClasses, int imgWidth, int imgHeight)
{
    float confThreshold = 0.8f; // Confidence threshold
    float nmsThreshold = 0.1f;  // Further reduce NMS threshold for more aggressive suppression

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    int imageHeight = image.rows;
    int imageWidth = image.cols;

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

        // Ensure box is within valid range
        int x_min = static_cast<int>(std::max((x_center - width / 2) * imageWidth, 0.0f));
        int y_min = static_cast<int>(std::max((y_center - height / 2) * imageHeight, 0.0f));
        int box_width = static_cast<int>(width * imageWidth);
        int box_height = static_cast<int>(height * imageHeight);

        float confidence = outputScores[i]; // Get confidence score
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

    // Perform Non-Maximum Suppression (NMS) with lower threshold
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

    // Debug: Print the number of boxes before and after NMS
    std::cout << "Number of boxes before NMS: " << boxes.size() << std::endl;
    std::cout << "Number of boxes after NMS: " << nmsIndices.size() << std::endl;

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

    // Resize the image and pad it to keep the aspect ratio (use 128 as padding value)
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));
    cv::Mat paddedImage(inputHeight, inputWidth, CV_32FC3, cv::Scalar(128, 128, 128));

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

    return inputTensorValues; // Tensor values in CHW format
}

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

    // Get both output names
    auto outputNameAllocated1 = session.GetOutputNameAllocated(0, allocator);
    auto outputNameAllocated2 = session.GetOutputNameAllocated(1, allocator);
    const char *outputName1 = outputNameAllocated1.get(); // Bounding boxes
    const char *outputName2 = outputNameAllocated2.get(); // Class scores

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

    // Run inference, expecting two outputs: one for boxes and one for class scores
    std::vector<const char *> inputNames = {inputName};
    std::vector<const char *> outputNames = {outputName1, outputName2}; // Bounding boxes and class scores
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 2);

    if (outputTensors.size() != 2)
    {
        std::cerr << "Expected 2 output tensors (boxes and scores), got " << outputTensors.size() << std::endl;
        return;
    }

    // Get outputs (bounding boxes and class scores)
    float *outputBoxes = outputTensors[0].GetTensorMutableData<float>();
    float *outputScores = outputTensors[1].GetTensorMutableData<float>();

    int numBoxes = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1]; // Number of detections

    std::vector<int> outputClasses(numBoxes);
    std::vector<float> confidences(numBoxes);

    int numClasses = 80; // Assuming COCO dataset

    // Get the class with the highest score
    for (int i = 0; i < numBoxes; i++)
    {
        float maxScore = 0.0f;
        int bestClass = -1;
        for (int j = 0; j < numClasses; j++)
        {
            float score = outputScores[i * numClasses + j];
            if (score > maxScore)
            {
                maxScore = score;
                bestClass = j;
            }
        }
        confidences[i] = maxScore;
        outputClasses[i] = bestClass;
    }

    // Post-process and save the image
    postProcessAndSaveImage(image, std::vector<float>(outputBoxes, outputBoxes + numBoxes * 4),
                            confidences, outputClasses, numBoxes, numClasses, inputWidth, inputHeight);
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
