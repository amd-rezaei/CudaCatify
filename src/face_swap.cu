#include "face_swap.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudawarping.hpp>

// CUDA-based face detection and emoji replacement
void detectAndReplaceFaces(cv::Mat &image, cv::Mat &catEmoji)
{
    // Convert to grayscale for face detection
    cv::cuda::GpuMat gpuImage, gpuGray, facesBuf;
    gpuImage.upload(image);
    cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);

    // Load the Haar Cascade face detector (assumes the file is in the current directory)
    cv::Ptr<cv::cuda::CascadeClassifier> faceCascade = cv::cuda::CascadeClassifier::create("haarcascade_frontalface_default.xml");

    // Detect faces
    faceCascade->detectMultiScale(gpuGray, facesBuf);

    // Convert detected faces back to CPU for further processing
    std::vector<cv::Rect> faces;
    faceCascade->convert(facesBuf, faces);

    // Replace faces with cat emoji
    for (const auto &face : faces)
    {
        // Resize the cat emoji to match the size of the detected face
        cv::Mat resizedCat;
        cv::resize(catEmoji, resizedCat, cv::Size(face.width, face.height));

        // Overlay the cat emoji on the original image at the detected face location
        for (int y = 0; y < resizedCat.rows; ++y)
        {
            for (int x = 0; x < resizedCat.cols; ++x)
            {
                cv::Vec4b catPixel = resizedCat.at<cv::Vec4b>(y, x); // Assuming catEmoji has an alpha channel
                if (catPixel[3] > 0)
                { // Check if the pixel is not transparent
                    image.at<cv::Vec3b>(face.y + y, face.x + x) = cv::Vec3b(catPixel[0], catPixel[1], catPixel[2]);
                }
            }
        }
    }
}
