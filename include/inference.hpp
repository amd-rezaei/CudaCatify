// inference.hpp
#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <opencv.hpp>
#include <string>
#include <vector>

// Function to run inference on a single image
std::vector<cv::Rect> runInference(const std::string &yolov5ModelPath, cv::Mat &image, float conf_thres, float iou_thres, std::vector<int> &classIds, int &imgWidth, int &imgHeight);

// Function to run inference on a video
void runInferenceVideo(const std::string &onnx_model, const std::string &video_path, const std::string &emoji_path, float conf_thres, float iou_thres, float blend_ratio, const std::string &output_dir);

#endif // INFERENCE_HPP
