#include <opencv2/opencv.hpp>
#include "face_swap.h"

int main()
{
    // Load the input image
    cv::Mat inputImage = cv::imread("data/input/input_image.jpg");
    if (inputImage.empty())
    {
        std::cerr << "Error: Could not load input image." << std::endl;
        return -1;
    }

    // Load the cat emoji image (assumed to have transparency)
    cv::Mat catEmoji = cv::imread("data/input/cat_emoji.png", cv::IMREAD_UNCHANGED);
    if (catEmoji.empty())
    {
        std::cerr << "Error: Could not load cat emoji image." << std::endl;
        return -1;
    }

    // Detect faces and replace them with the cat emoji using CUDA
    detectAndReplaceFaces(inputImage, catEmoji);

    // Save the output image
    cv::imwrite("data/output/output_image.jpg", inputImage);

    std::cout << "Processing completed. Check the output in data/output/output_image.jpg" << std::endl;
    return 0;
}
