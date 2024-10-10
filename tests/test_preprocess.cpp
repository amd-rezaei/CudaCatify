// test_preprocess.cpp
#include "preprocess.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(Preprocess, ImageResizing)
{
    cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3); // Create a black 100x100 image
    int inputWidth = 640;
    int inputHeight = 640;

    std::vector<float> processedImage = preprocessImage(image, inputWidth, inputHeight);

    // Check the size of the processed image tensor (3 channels * 640 * 640)
    EXPECT_EQ(processedImage.size(), 3 * inputWidth * inputHeight);
}
