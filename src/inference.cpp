// inference.cpp
#include "inference.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "nms.hpp"
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <iostream>

// Function to run inference on a single image
std::vector<cv::Rect> runInference(const std::string &yolov5ModelPath, cv::Mat &image, float conf_thres, float iou_thres, std::vector<int> &classIds, int &imgWidth, int &imgHeight)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    Ort::SessionOptions sessionOptions;
    Ort::Session yolov5Session(env, yolov5ModelPath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    if (image.empty())
    {
        std::cerr << "Error: Image is empty!" << std::endl;
        return {};
    }

    auto yolov5InputShape = yolov5Session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<float> yolov5InputValues = preprocessImage(image, yolov5InputShape[3], yolov5InputShape[2]);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value yolov5InputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, yolov5InputValues.data(), yolov5InputValues.size(), yolov5InputShape.data(), yolov5InputShape.size());

    auto yolov5InputNameAllocated = yolov5Session.GetInputNameAllocated(0, allocator);
    const char *yolov5InputName = yolov5InputNameAllocated.get();
    auto yolov5OutputNameAllocated = yolov5Session.GetOutputNameAllocated(0, allocator);
    const char *yolov5OutputName = yolov5OutputNameAllocated.get();

    std::vector<const char *> yolov5InputNames = {yolov5InputName};
    std::vector<const char *> yolov5OutputNames = {yolov5OutputName};
    auto yolov5OutputTensors = yolov5Session.Run(Ort::RunOptions{nullptr}, yolov5InputNames.data(), &yolov5InputTensor, 1, yolov5OutputNames.data(), 1);

    const float *yolov5OutputData = yolov5OutputTensors[0].GetTensorMutableData<float>();
    size_t outputSize = yolov5OutputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<float> outputData(yolov5OutputData, yolov5OutputData + outputSize);

    return postProcessAndReturnBoxes(image, outputData, yolov5InputShape[3], yolov5InputShape[2], conf_thres, iou_thres, classIds);
}

// Function to run inference on a video
void runInferenceVideo(const std::string &onnx_model, const std::string &video_path, const std::string &emoji_path, float conf_thres, float iou_thres, float blend_ratio, const std::string &output_dir)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file!" << std::endl;
        return;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    std::string baseName = std::filesystem::path(video_path).stem().string();
    std::string output_file = output_dir + "/" + baseName + "_private.avi";
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty())
        {
            std::cerr << "Error: Could not read frame!" << std::endl;
            break;
        }

        std::vector<int> classIds;
        int imgWidth = frame.cols;
        int imgHeight = frame.rows;
        std::vector<cv::Rect> scaled_boxes = runInference(onnx_model, frame, conf_thres, iou_thres, classIds, imgWidth, imgHeight);
        replaceWithEmojiInPostProcess(frame, scaled_boxes, emoji_path, blend_ratio);

        writer.write(frame);
    }

    cap.release();
    writer.release();
    std::cout << "Processed video saved as: " << output_file << std::endl;
}
