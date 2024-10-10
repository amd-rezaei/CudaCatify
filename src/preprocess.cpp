// preprocess.cpp
#include "preprocess.hpp"


// Preprocess image for ONNX input using OpenCV and NPP
std::vector<float> preprocessImage(cv::Mat &image, int inputWidth, int inputHeight)
{
    if (image.empty())
    {
        return {};
    }

    int originalHeight = image.rows;
    int originalWidth = image.cols;

    // Step 1: Resize and pad image using OpenCV
    float scale = std::min(float(inputWidth) / originalWidth, float(inputHeight) / originalHeight);
    int newWidth = static_cast<int>(originalWidth * scale);
    int newHeight = static_cast<int>(originalHeight * scale);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

    cv::Mat paddedImage(inputHeight, inputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    resizedImage.copyTo(paddedImage(cv::Rect((inputWidth - newWidth) / 2, (inputHeight - newHeight) / 2, newWidth, newHeight)));

    // Step 2: Use NPP for image conversion and normalization

    // Allocate GPU memory for input and output images
    Npp8u *d_inputImage = nullptr;
    Npp32f *d_outputImage = nullptr;

    size_t inputImageSize = paddedImage.total() * paddedImage.elemSize();
    size_t outputImageSize = inputWidth * inputHeight * 3 * sizeof(Npp32f);

    cudaMalloc(&d_inputImage, inputImageSize);
    cudaMalloc(&d_outputImage, outputImageSize);

    cudaMemcpy(d_inputImage, paddedImage.data, inputImageSize, cudaMemcpyHostToDevice);

    NppiSize srcSize = {inputWidth, inputHeight};
    nppiConvert_8u32f_C3R(d_inputImage, inputWidth * 3, d_outputImage, inputWidth * 3 * sizeof(Npp32f), srcSize);

    Npp32f divConstants[3] = {255.0f, 255.0f, 255.0f};
    nppiDivC_32f_C3IR(divConstants, d_outputImage, inputWidth * 3 * sizeof(Npp32f), srcSize);

    std::vector<float> inputTensorValues(outputImageSize / sizeof(Npp32f));
    cudaMemcpy(inputTensorValues.data(), d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Step 5: Convert the image data to CHW format
    std::vector<float> chwTensorValues;
    chwTensorValues.reserve(outputImageSize);

    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < inputHeight; ++h)
        {
            for (int w = 0; w < inputWidth; ++w)
            {
                chwTensorValues.push_back(inputTensorValues[h * inputWidth * 3 + w * 3 + c]);
            }
        }
    }

    return chwTensorValues;
}

// Scale bounding boxes back to the original image size
cv::Rect scaleBoundingBox(cv::Rect box, int originalWidth, int originalHeight, int inputWidth, int inputHeight, float scale)
{
    int x_offset = (inputWidth - static_cast<int>(originalWidth * scale)) / 2;
    int y_offset = (inputHeight - static_cast<int>(originalHeight * scale)) / 2;

    int x_min = static_cast<int>((box.x - x_offset) / scale);
    int y_min = static_cast<int>((box.y - y_offset) / scale);
    int width = static_cast<int>(box.width / scale);
    int height = static_cast<int>(box.height / scale);

    return cv::Rect(x_min, y_min, width, height);
}
