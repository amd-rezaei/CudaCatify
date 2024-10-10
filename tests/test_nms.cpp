// test_nms.cpp
#include "nms.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(NMS, SimpleNMS)
{
    std::vector<cv::Rect> boxes = {
        cv::Rect(100, 100, 50, 50),
        cv::Rect(110, 110, 50, 50),
        cv::Rect(200, 200, 50, 50)};

    std::vector<float> confidences = {0.9f, 0.8f, 0.95f};
    std::vector<int> class_ids = {0, 0, 0};
    float iou_thres = 0.3f; // Lower the threshold to 0.3

    std::vector<int> indices = performNMS(boxes, confidences, class_ids, iou_thres);

    // We expect only two boxes to remain after NMS
    EXPECT_EQ(indices.size(), 2);
}
