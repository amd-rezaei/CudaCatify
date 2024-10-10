// preprocess.hpp
#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// Preprocess an image for ONNX input
std::vector<float> preprocessImage(cv::Mat &image, int inputWidth, int inputHeight);

// Adjust bounding boxes to match original image size
cv::Rect scaleBoundingBox(cv::Rect box, int originalWidth, int originalHeight, int inputWidth, int inputHeight, float scale);

#endif // PREPROCESS_HPP
