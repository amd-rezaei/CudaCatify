#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime header
#include <cmath>
#include <algorithm>
#include <numeric> // Add this for std::iota
#include <fstream>

// Helper function to apply sigmoid
float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// Function to clamp values
float clamp(float value, float minValue, float maxValue)
{
    return std::max(minValue, std::min(value, maxValue));
}

// Convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
cv::Rect xywh2xyxy(float x_center, float y_center, float width, float height)
{
    int x_min = static_cast<int>(x_center - (width / 2));
    int y_min = static_cast<int>(y_center - (height / 2));
    int x_max = static_cast<int>(x_center + (width / 2));
    int y_max = static_cast<int>(y_center + (height / 2));
    return cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
}

// Perform Non-Maximum Suppression (NMS)
std::vector<int> performNMS(std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &class_ids, float iou_thres)
{
    std::vector<int> indices;
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0); // Fill indices with 0, 1, ..., n-1

    // Sort by confidence scores
    std::sort(idxs.begin(), idxs.end(), [&confidences](int i1, int i2)
              { return confidences[i1] > confidences[i2]; });

    while (!idxs.empty())
    {
        int idx = idxs[0];
        indices.push_back(idx);
        std::vector<int> newIdxs;

        for (size_t i = 1; i < idxs.size(); ++i)
        {
            int currentIdx = idxs[i];

            // Calculate the IoU (Intersection over Union)
            float interArea = (boxes[idx] & boxes[currentIdx]).area();
            float unionArea = boxes[idx].area() + boxes[currentIdx].area() - interArea;
            float iou = interArea / unionArea;

            // Keep boxes that don't overlap too much (IoU < threshold)
            if (iou <= iou_thres)
            {
                newIdxs.push_back(currentIdx);
            }
        }

        idxs = std::move(newIdxs);
    }

    return indices;
}

// Non-Maximum Suppression (NMS) for YOLOv5 detection output
std::vector<int> non_max_suppression_face(const std::vector<float> &output, std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &class_ids, float conf_thres, float iou_thres, int num_classes)
{
    int numBoxes = output.size() / 16; // Assuming each box has 16 values (x, y, w, h, obj_conf, class_probs, etc.)

    // Open CSV file to log results after multiplying class confidence with object confidence
    std::ofstream csvFile("after_conf_cls_multiplication_cpp.csv");
    if (!csvFile.is_open())
    {
        std::cerr << "Error: Could not open after_conf_cls_multiplication_cpp.csv for writing." << std::endl;
        return {};
    }

    // Write CSV header
    csvFile << "x_center,y_center,width,height,confidence";
    for (int j = 0; j < 10; ++j) // Assuming 10 classes
    {
        csvFile << ",Class_" << j << "_Probability";
    }
    csvFile << std::endl;

    // Iterate over the predictions
    for (int i = 0; i < numBoxes; ++i)
    {
        int index = i * 16;

        // Object confidence
        float confidence = output[index + 4];

        if (confidence >= conf_thres)
        {
            // Extract box coordinates and dimensions (center x, center y, width, height)
            float x_center = output[index];     // Center x (normalized)
            float y_center = output[index + 1]; // Center y (normalized)
            float width = output[index + 2];    // Width (normalized)
            float height = output[index + 3];   // Height (normalized)

            // Store box coordinates (in pixels)
            boxes.push_back(xywh2xyxy(x_center, y_center, width, height));

            // Store confidence score
            confidences.push_back(confidence);

            // Write raw data to the CSV after multiplying class confidence with object confidence
            csvFile << x_center << "," << y_center << "," << width << "," << height << "," << confidence;

            // Determine best class
            float max_class_score = -1.0f;
            int best_class = -1;
            for (int j = 0; j < num_classes; ++j)
            {
                // Multiply class confidence with object confidence
                float class_prob = output[index + 5 + j] * confidence;

                // Write to CSV
                csvFile << "," << class_prob;

                if (class_prob > max_class_score)
                {
                    best_class = j;
                    max_class_score = class_prob;
                }
            }
            class_ids.push_back(best_class);
            csvFile << std::endl;
        }
    }

    // Close CSV file
    csvFile.close();
    std::cout << "Confidence-class multiplication data written to after_conf_cls_multiplication_cpp.csv." << std::endl;

    // Apply Non-Maximum Suppression (NMS)
    return performNMS(boxes, confidences, class_ids, iou_thres);
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

    std::cout << "Image preprocessed to CHW format." << std::endl;
    return inputTensorValues; // Tensor values in CHW format
}

// Post-process and save the image with bounding boxes and write NMS results to CSV
void postProcessAndSaveImage(cv::Mat &image, const std::vector<float> &output, int imgWidth, int imgHeight, float conf_thres, float iou_thres)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // Perform NMS after processing output data
    std::vector<int> nms_indices = non_max_suppression_face(output, boxes, confidences, classIds, conf_thres, iou_thres, 10);

    // Open CSV file for NMS results
    std::ofstream nmsCsvFile("nms_results_cpp.csv");
    if (!nmsCsvFile.is_open())
    {
        std::cerr << "Error: Could not open nms_results_cpp.csv for writing." << std::endl;
        return;
    }

    // Write header for NMS results
    nmsCsvFile << "Box_X,Box_Y,Box_Width,Box_Height,Confidence,Class_ID" << std::endl;

    // Log and draw bounding boxes after NMS
    for (int idx : nms_indices)
    {
        cv::Rect box = boxes[idx];
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class: " + std::to_string(classIds[idx]) + " Confidence: " + std::to_string(confidences[idx]);
        cv::putText(image, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // Write NMS results to the CSV file
        nmsCsvFile << box.x << "," << box.y << "," << box.width << "," << box.height << ","
                   << confidences[idx] << "," << classIds[idx] << std::endl;

        std::cout << "Bounding Box " << idx << ": " << box.x << ", " << box.y << ", " << box.width << ", " << box.height
                  << " Confidence: " << confidences[idx] << " Class: " << classIds[idx] << std::endl;
    }

    nmsCsvFile.close();
    cv::imwrite("output_with_bboxes.jpg", image);
    std::cout << "Output image saved as 'output_with_bboxes.jpg'." << std::endl;
}

// Run inference using YOLOv5 model
void runInference(const std::string &yolov5ModelPath, const std::string &imagePath, float conf_thres, float iou_thres)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    Ort::SessionOptions sessionOptions;
    Ort::Session yolov5Session(env, yolov5ModelPath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    // Load input image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return;
    }

    // Get YOLOv5 input info
    auto yolov5InputShape = yolov5Session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<float> yolov5InputValues = preprocessImage(image, yolov5InputShape[3], yolov5InputShape[2]);

    // Create input tensor for YOLOv5
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value yolov5InputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, yolov5InputValues.data(), yolov5InputValues.size(), yolov5InputShape.data(), yolov5InputShape.size());

    // Allocate memory for YOLOv5 input/output names
    auto yolov5InputNameAllocated = yolov5Session.GetInputNameAllocated(0, allocator);
    const char *yolov5InputName = yolov5InputNameAllocated.get();
    auto yolov5OutputNameAllocated = yolov5Session.GetOutputNameAllocated(0, allocator);
    const char *yolov5OutputName = yolov5OutputNameAllocated.get();

    // Log the start of inference
    std::cout << "Running YOLOv5 inference..." << std::endl;

    // Run YOLOv5 inference
    std::vector<const char *> yolov5InputNames = {yolov5InputName};
    std::vector<const char *> yolov5OutputNames = {yolov5OutputName};
    auto yolov5OutputTensors = yolov5Session.Run(Ort::RunOptions{nullptr}, yolov5InputNames.data(), &yolov5InputTensor, 1, yolov5OutputNames.data(), 1);

    // Log after inference completion
    std::cout << "YOLOv5 inference completed." << std::endl;

    // Access tensor data correctly for YOLOv5 output
    const float *yolov5OutputData = yolov5OutputTensors[0].GetTensorMutableData<float>();
    size_t outputSize = yolov5OutputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    // Store the output data in a vector
    std::vector<float> outputData(yolov5OutputData, yolov5OutputData + outputSize);

    // Log output size
    std::cout << "Output size: " << outputSize << std::endl;

    // Post-process output and save
    postProcessAndSaveImage(image, outputData, yolov5InputShape[3], yolov5InputShape[2], conf_thres, iou_thres);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <yolov5_model.onnx> <image> <conf_thres> <iou_thres>" << std::endl;
        return -1;
    }

    float conf_thres = std::stof(argv[3]);
    float iou_thres = std::stof(argv[4]);
    runInference(argv[1], argv[2], conf_thres, iou_thres);

    return 0;
}
