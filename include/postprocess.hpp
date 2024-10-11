// post_process.hpp
#ifndef POST_PROCESS_HPP
#define POST_PROCESS_HPP

#include "nms.hpp"
#include "preprocess.hpp"
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Replace bounding boxes with emoji image
void replaceWithEmojiInPostProcess(cv::Mat &image, const std::vector<cv::Rect> &boxes, const std::string &emojiPath, float blendRatio);

std::vector<cv::Rect> postProcessAndReturnBoxes(cv::Mat &image, const std::vector<float> &output, int imgWidth, int imgHeight, float conf_thres, float iou_thres, std::vector<int> &classIds);

#endif // POST_PROCESS_HPP
