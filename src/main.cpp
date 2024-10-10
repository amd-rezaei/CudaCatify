#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime header
#include <cmath>
#include <algorithm>
#include <numeric>
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cxxopts.hpp>
#include <filesystem> // For std::filesystem::path

std::string getBaseName(const std::string &filepath)
{
    // Use std::filesystem to extract the filename without extension
    std::filesystem::path p(filepath);
    return p.stem().string(); // Returns the filename without the extension
}

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

// Preprocess image for ONNX input using OpenCV for resizing and padding, then NPP for conversion, normalization, and CHW reformat
std::vector<float> preprocessImage(cv::Mat &image, int inputWidth, int inputHeight)
{
    if (image.empty())
    {
        return {};
    }

    int originalHeight = image.rows;
    int originalWidth = image.cols;

    // Step 1: Resize and pad image using OpenCV
    float scale = std::min(float(inputWidth) / originalWidth, float(inputHeight) / originalHeight);
    int newWidth = static_cast<int>(originalWidth * scale);
    int newHeight = static_cast<int>(originalHeight * scale);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

    // Pad the resized image to fit the target size
    cv::Mat paddedImage(inputHeight, inputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    resizedImage.copyTo(paddedImage(cv::Rect((inputWidth - newWidth) / 2, (inputHeight - newHeight) / 2, newWidth, newHeight)));

    // Step 2: Use NPP for image conversion and normalization

    // Allocate GPU memory for input and output images
    Npp8u *d_inputImage = nullptr;
    Npp32f *d_outputImage = nullptr;

    size_t inputImageSize = paddedImage.total() * paddedImage.elemSize();
    size_t outputImageSize = inputWidth * inputHeight * 3 * sizeof(Npp32f); // 3 channels, float32 format

    // Allocate memory on GPU for input and output
    cudaMalloc(&d_inputImage, inputImageSize);
    cudaMalloc(&d_outputImage, outputImageSize);

    // Copy the padded image from host to device
    cudaMemcpy(d_inputImage, paddedImage.data, inputImageSize, cudaMemcpyHostToDevice);

    // Convert image from Npp8u (uint8) to Npp32f (float32)
    NppiSize srcSize = {inputWidth, inputHeight};
    NppStatus nppStatus = nppiConvert_8u32f_C3R(d_inputImage, inputWidth * 3, d_outputImage, inputWidth * 3 * sizeof(Npp32f), srcSize);
    if (nppStatus != NPP_SUCCESS)
    {
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return {};
    }

    // Step 3: Normalize the image (0-255 to 0-1) using NPP
    Npp32f divConstants[3] = {255.0f, 255.0f, 255.0f};
    nppStatus = nppiDivC_32f_C3IR(divConstants, d_outputImage, inputWidth * 3 * sizeof(Npp32f), srcSize);
    if (nppStatus != NPP_SUCCESS)
    {
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return {};
    }

    // Step 4: Copy data back to host after normalization
    std::vector<float> inputTensorValues(outputImageSize / sizeof(Npp32f));
    cudaMemcpy(inputTensorValues.data(), d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Step 5: Convert the image data to CHW format (Channels-First) expected by YOLO model
    std::vector<float> chwTensorValues;
    chwTensorValues.reserve(outputImageSize); // Pre-allocate memory

    // Copy data channel by channel (R, G, B)
    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < inputHeight; ++h)
        {
            for (int w = 0; w < inputWidth; ++w)
            {
                chwTensorValues.push_back(inputTensorValues[h * inputWidth * 3 + w * 3 + c]);
            }
        }
    }

    // Return the tensor in CHW format
    return chwTensorValues;
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
    cv::imwrite("./data/output/output_with_emojis.jpg", image);
    std::cout << "Output image with emojis saved as 'output_with_emojis.jpg'." << std::endl;
}

// Function to replace bounding boxes with emoji image using NPP for resizing and applying blending
void replaceWithEmojiInPostProcessNPP(cv::Mat &image, const std::vector<cv::Rect> &boxes, const std::string &emojiPath, float blendRatio)
{
    // Load the emoji image
    cv::Mat emoji = cv::imread(emojiPath, cv::IMREAD_UNCHANGED); // Load with alpha channel if available
    if (emoji.empty())
    {
        std::cerr << "Error: Could not load emoji image!" << std::endl;
        return;
    }

    // Check that the emoji has 4 channels (RGBA), otherwise add alpha channel
    if (emoji.channels() == 3)
    {
        cv::cvtColor(emoji, emoji, cv::COLOR_BGR2BGRA);
    }

    // Ensure blendRatio is within valid range (0.0 to 1.0)
    blendRatio = std::clamp(blendRatio, 0.0f, 1.0f);

    // Replace each bounding box with the emoji using NPP for resizing
    for (const auto &box : boxes)
    {
        std::cout << "Scaled Bounding Box: [x=" << box.x << ", y=" << box.y
                  << ", width=" << box.width << ", height=" << box.height << "]" << std::endl;

        // Resize the emoji to fit the bounding box using NPP
        NppiSize srcSize = {emoji.cols, emoji.rows};
        NppiSize dstSize = {box.width, box.height};

        // Define steps (strides), which are the number of bytes per row
        int srcStep = emoji.step;
        int dstStep = box.width * 4; // 4 channels (RGBA)

        // Allocate memory on the GPU for the source and destination
        Npp8u *d_src = nullptr, *d_dst = nullptr;
        cudaMalloc(&d_src, emoji.total() * emoji.elemSize());
        cudaMalloc(&d_dst, dstStep * box.height);

        // Copy the emoji to the device
        cudaMemcpy(d_src, emoji.data, emoji.total() * emoji.elemSize(), cudaMemcpyHostToDevice);

        // Perform NPP-based resizing
        NppStatus status = nppiResize_8u_C4R(d_src, srcStep, srcSize, {0, 0, srcSize.width, srcSize.height}, d_dst, dstStep, dstSize, {0, 0, dstSize.width, dstSize.height}, NPPI_INTER_LINEAR);

        if (status != NPP_SUCCESS)
        {
            std::cerr << "Error: NPP resize failed with error code: " << status << std::endl;
            cudaFree(d_src);
            cudaFree(d_dst);
            return;
        }

        // Copy resized emoji back to the host
        cv::Mat resized_emoji(box.height, box.width, CV_8UC4);
        cudaMemcpy(resized_emoji.data, d_dst, dstStep * box.height, cudaMemcpyDeviceToHost);

        // Ensure the emoji fits inside the image
        if (box.x >= 0 && box.y >= 0 && (box.x + box.width <= image.cols) && (box.y + box.height <= image.rows))
        {
            // Region of interest (ROI) in the original image
            cv::Mat roi = image(box);

            // Handle transparency by blending the emoji with the ROI
            for (int y = 0; y < resized_emoji.rows; ++y)
            {
                for (int x = 0; x < resized_emoji.cols; ++x)
                {
                    cv::Vec4b &emoji_pixel = resized_emoji.at<cv::Vec4b>(y, x);
                    if (emoji_pixel[3] > 0) // If alpha > 0, blend pixel
                    {
                        cv::Vec3b &roi_pixel = roi.at<cv::Vec3b>(y, x);
                        roi_pixel = cv::Vec3b(
                            static_cast<uchar>(roi_pixel[0] * (1 - blendRatio) + emoji_pixel[0] * blendRatio),
                            static_cast<uchar>(roi_pixel[1] * (1 - blendRatio) + emoji_pixel[1] * blendRatio),
                            static_cast<uchar>(roi_pixel[2] * (1 - blendRatio) + emoji_pixel[2] * blendRatio));
                    }
                }
            }
        }

        // Free GPU memory
        cudaFree(d_src);
        cudaFree(d_dst);
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
    cxxopts::Options options("cudacatify", "A description of cudacatify");

    // Define the available options (no onnx_model or jpg_photo here)
    options.add_options()("emoji", "Path to the emoji image", cxxopts::value<std::string>()->default_value("./data/input/kitty_emoji.png"))("conf_thres", "Confidence threshold", cxxopts::value<float>()->default_value("0.5"))("iou_thresh", "IoU threshold", cxxopts::value<float>()->default_value("0.5"))("blend_thresh", "Blend threshold", cxxopts::value<float>()->default_value("1.0"))("output_dir", "Output directory", cxxopts::value<std::string>()->default_value("./data/output/"))("h,help", "Print usage");

    // Mark 'onnx_model' and 'jpg_photo' as positional arguments
    options.add_options()("onnx_model", "Path to the ONNX model", cxxopts::value<std::string>())("jpg_photo", "Path to the input image", cxxopts::value<std::string>());

    options.parse_positional({"onnx_model", "jpg_photo"});
    options.positional_help("<onnx_model> <jpg_photo>").show_positional_help();

    // Parse the arguments
    auto result = options.parse(argc, argv);

    // Handle help
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Required positional arguments
    if (!result.count("onnx_model") || !result.count("jpg_photo"))
    {
        std::cerr << "Error: <onnx_model> and <jpg_photo> are required." << std::endl;
        std::cout << options.help() << std::endl;
        return -1;
    }

    // Accessing positional arguments
    std::string onnx_model = result["onnx_model"].as<std::string>();
    std::string jpg_photo = result["jpg_photo"].as<std::string>();

    // Optional arguments with default values
    std::string optional_emoji = result["emoji"].as<std::string>();
    float optional_conf_thres = result["conf_thres"].as<float>();
    float optional_iou_thresh = result["iou_thresh"].as<float>();
    float optional_blend_threshold = result["blend_thresh"].as<float>();
    std::string optional_output_dir = result["output_dir"].as<std::string>();

    // Load the image
    cv::Mat image = cv::imread(jpg_photo);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Run inference to get scaled bounding boxes and class IDs
    std::vector<int> classIds;
    int imgWidth = image.cols;
    int imgHeight = image.rows;
    std::vector<cv::Rect> scaled_boxes = runInference(onnx_model, jpg_photo, optional_conf_thres, optional_iou_thresh, classIds, imgWidth, imgHeight);

    // Replace detected bounding boxes with the emoji image, using the blend ratio
    replaceWithEmojiInPostProcessNPP(image, scaled_boxes, optional_emoji, optional_blend_threshold);

    // Generate output filename based on the input image
    std::string baseName = getBaseName(jpg_photo); // Extract the base name of the input image
    std::string output_file = optional_output_dir + "/" + baseName + "_private.jpg";

    // Ensure the output directory exists
    std::filesystem::create_directories(optional_output_dir);

    // Save the output image
    cv::imwrite(output_file, image);
    std::cout << "Output image saved as: " << output_file << std::endl;

    return 0;
}