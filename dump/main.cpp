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
void postProcessAndSaveImage(cv::Mat &image, const std::vector<float> &outputBoxes, const std::vector<float> &outputScores, const std::vector<int> &outputClasses, int numBoxes, int numClasses)
{
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    int imgHeight = image.rows;
    int imgWidth = image.cols;

    // Iterate over the number of detections
    for (int i = 0; i < numBoxes; ++i)
    {
        float x_center = outputBoxes[i * 4];
        float y_center = outputBoxes[i * 4 + 1];
        float width = outputBoxes[i * 4 + 2];
        float height = outputBoxes[i * 4 + 3];

        int x_min = static_cast<int>((x_center - width / 2) * imgWidth);
        int y_min = static_cast<int>((y_center - height / 2) * imgHeight);
        int box_width = static_cast<int>(width * imgWidth);
        int box_height = static_cast<int>(height * imgHeight);

        float confidence = outputScores[i];
        int classId = outputClasses[i];

        // Only consider boxes with high enough confidence
        if (confidence > confThreshold)
        {
            boxes.emplace_back(x_min, y_min, box_width, box_height);
            confidences.push_back(confidence);
            classIds.push_back(classId);

            // Print out the details of the detected objects
            std::cout << "Detected object: Class " << classId
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

        // Put label with class ID and confidence
        std::string label = "Class " + std::to_string(classId) + ": " + std::to_string(confidence);
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
    cv::resize(image, resizedImage, cv::Size(inputWidth, inputHeight));
    resizedImage.convertTo(resizedImage, CV_32F, 1 / 255.0);

    // Convert HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(resizedImage, channels);

    std::vector<float> inputTensorValues;
    for (int c = 0; c < 3; ++c)
    {
        inputTensorValues.insert(inputTensorValues.end(), (float *)channels[c].datastart, (float *)channels[c].dataend);
    }

    return inputTensorValues;
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
