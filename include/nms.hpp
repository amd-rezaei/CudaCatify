// nms.hpp
#ifndef NMS_HPP
#define NMS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

// Function to perform Non-Maximum Suppression (NMS)
std::vector<int> performNMS(std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &class_ids, float iou_thres);

// Function for processing YOLOv5 output and applying NMS
std::vector<int> non_max_suppression_face(const std::vector<float> &output, std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &class_ids, float conf_thres, float iou_thres, int num_classes);

#endif // NMS_HPP
