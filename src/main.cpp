#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime header
#include <cmath>
#include <algorithm>
#include <numeric>

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

            // Determine best class
            float max_class_score = -1.0f;
            int best_class = -1;
            for (int j = 0; j < num_classes; ++j)
            {
                float class_prob = output[index + 5 + j] * confidence;
                if (class_prob > max_class_score)
                {
                    best_class = j;
                    max_class_score = class_prob;
                }
            }
            class_ids.push_back(best_class);
        }
    }

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

// Adjust bounding boxes to match original image size
cv::Rect scaleBoundingBox(cv::Rect box, int originalWidth, int originalHeight, int inputWidth, int inputHeight, float scale)
{
    int x_offset = (inputWidth - static_cast<int>(originalWidth * scale)) / 2;
    int y_offset = (inputHeight - static_cast<int>(originalHeight * scale)) / 2;

    // Scale the box dimensions back to the original image size
    int x_min = static_cast<int>((box.x - x_offset) / scale);
    int y_min = static_cast<int>((box.y - y_offset) / scale);
    int width = static_cast<int>(box.width / scale);
    int height = static_cast<int>(box.height / scale);

    return cv::Rect(x_min, y_min, width, height);
}

// Post-process the results and return scaled bounding boxes
std::vector<cv::Rect> postProcessAndReturnBoxes(cv::Mat &image, const std::vector<float> &output, int imgWidth, int imgHeight, float conf_thres, float iou_thres, std::vector<int> &classIds)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    // Perform NMS after processing output data
    std::vector<int> nms_indices = non_max_suppression_face(output, boxes, confidences, classIds, conf_thres, iou_thres, 10);

    // Calculate scale factor used during preprocessing
    float scale = std::min(float(imgWidth) / image.cols, float(imgHeight) / image.rows);

    // Scale the bounding boxes to match the original image size
    std::vector<cv::Rect> scaled_boxes;
    for (int idx : nms_indices)
    {
        cv::Rect scaled_box = scaleBoundingBox(boxes[idx], image.cols, image.rows, imgWidth, imgHeight, scale);
        scaled_boxes.push_back(scaled_box);
    }

    return scaled_boxes;
}

// Function to replace bounding boxes with emoji image
void replaceWithEmojiInPostProcess(cv::Mat &image, const std::vector<cv::Rect> &boxes, const std::string &emojiPath)
{
    // Load the emoji image
    cv::Mat emoji = cv::imread(emojiPath, cv::IMREAD_UNCHANGED); // Load with alpha channel if available
    if (emoji.empty())
    {
        std::cerr << "Error: Could not load emoji image!" << std::endl;
        return;
    }

    // Replace each bounding box with the emoji
    for (const auto &box : boxes)
    {
        // Print out the scaled bounding box details
        std::cout << "Scaled Bounding Box: [x=" << box.x << ", y=" << box.y
                  << ", width=" << box.width << ", height=" << box.height << "]" << std::endl;

        // Resize the emoji to fit the bounding box
        cv::Mat resized_emoji;
        cv::resize(emoji, resized_emoji, cv::Size(box.width, box.height));

        // Ensure the emoji fits inside the image
        if (box.x >= 0 && box.y >= 0 && (box.x + box.width <= image.cols) && (box.y + box.height <= image.rows))
        {
            // Region of interest (ROI) in the original image
            cv::Mat roi = image(box);

            // Handle transparency if emoji has 4 channels (RGBA)
            if (resized_emoji.channels() == 4)
            {
                for (int y = 0; y < resized_emoji.rows; ++y)
                {
                    for (int x = 0; x < resized_emoji.cols; ++x)
                    {
                        cv::Vec4b &emoji_pixel = resized_emoji.at<cv::Vec4b>(y, x);
                        if (emoji_pixel[3] > 0) // If alpha > 0, replace pixel
                        {
                            roi.at<cv::Vec3b>(y, x) = cv::Vec3b(emoji_pixel[0], emoji_pixel[1], emoji_pixel[2]); // Copy RGB
                        }
                    }
                }
            }
            else
            {
                // If no alpha channel, simply copy the resized emoji to the region
                resized_emoji.copyTo(roi);
            }
        }
    }

    // Save the output image with emojis
    cv::imwrite("output_with_emojis.jpg", image);
    std::cout << "Output image with emojis saved as 'output_with_emojis.jpg'." << std::endl;
}

// Run inference using YOLOv5 model
std::vector<cv::Rect> runInference(const std::string &yolov5ModelPath, const std::string &imagePath, float conf_thres, float iou_thres, std::vector<int> &classIds, int &imgWidth, int &imgHeight)
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
        return {};
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

    // Post-process output and return scaled bounding boxes
    return postProcessAndReturnBoxes(image, outputData, yolov5InputShape[3], yolov5InputShape[2], conf_thres, iou_thres, classIds);
}

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] << " <yolov5_model.onnx> <image> <conf_thres> <iou_thres> <emoji.jpg>" << std::endl;
        return -1;
    }

    float conf_thres = std::stof(argv[3]);
    float iou_thres = std::stof(argv[4]);

    // Load the image
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Run inference to get scaled bounding boxes and class IDs
    std::vector<int> classIds;
    int imgWidth = image.cols;
    int imgHeight = image.rows;
    std::vector<cv::Rect> scaled_boxes = runInference(argv[1], argv[2], conf_thres, iou_thres, classIds, imgWidth, imgHeight);

    // Replace detected bounding boxes with the emoji image
    replaceWithEmojiInPostProcess(image, scaled_boxes, argv[5]);

    return 0;
}
