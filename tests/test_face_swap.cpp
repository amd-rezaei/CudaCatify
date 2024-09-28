#include "face_swap.h"
#include <opencv2/opencv.hpp>
#include <cassert>

// Simple test case for detecting and replacing faces
void testFaceSwap()
{
    // Load a test image and cat emoji
    cv::Mat testImage = cv::imread("data/input/test_image.jpg");
    cv::Mat catEmoji = cv::imread("data/input/cat_emoji.png", cv::IMREAD_UNCHANGED);

    // Ensure that the images are loaded properly
    assert(!testImage.empty());
    assert(!catEmoji.empty());

    // Call the function to replace faces with cat emoji
    detectAndReplaceFaces(testImage, catEmoji);

    // Save the output to verify visually
    cv::imwrite("results/test_output.jpg", testImage);

    std::cout << "Test completed. Check the output in results/test_output.jpg" << std::endl;
}

int main()
{
    testFaceSwap();
    return 0;
}
