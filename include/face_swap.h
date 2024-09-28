#ifndef FACE_SWAP_H
#define FACE_SWAP_H

#include <opencv2/opencv.hpp>

// CUDA face detection and replacement function declarations
void detectAndReplaceFaces(cv::Mat &image, cv::Mat &catEmoji);

#endif // FACE_SWAP_H
