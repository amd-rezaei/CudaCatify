// post_process.cpp
#include "postprocess.hpp"


void replaceWithEmojiInPostProcess(cv::Mat &image, const std::vector<cv::Rect> &boxes, const std::string &emojiPath, float blendRatio)
{
    cv::Mat emoji = cv::imread(emojiPath, cv::IMREAD_UNCHANGED);
    if (emoji.empty())
    {
        std::cerr << "Error: Could not load emoji image!" << std::endl;
        return;
    }

    if (emoji.channels() == 3)
    {
        cv::cvtColor(emoji, emoji, cv::COLOR_BGR2BGRA);
    }

    blendRatio = std::clamp(blendRatio, 0.0f, 1.0f);

    for (const auto &box : boxes)
    {
        NppiSize srcSize = {emoji.cols, emoji.rows};
        NppiSize dstSize = {box.width, box.height};

        int srcStep = emoji.step;
        int dstStep = box.width * 4;

        Npp8u *d_src = nullptr, *d_dst = nullptr;
        cudaMalloc(&d_src, emoji.total() * emoji.elemSize());
        cudaMalloc(&d_dst, dstStep * box.height);

        cudaMemcpy(d_src, emoji.data, emoji.total() * emoji.elemSize(), cudaMemcpyHostToDevice);

        NppStatus status = nppiResize_8u_C4R(d_src, srcStep, srcSize, {0, 0, srcSize.width, srcSize.height}, d_dst, dstStep, dstSize, {0, 0, dstSize.width, dstSize.height}, NPPI_INTER_LINEAR);

        if (status != NPP_SUCCESS)
        {
            std::cerr << "Error: NPP resize failed with error code: " << status << std::endl;
            cudaFree(d_src);
            cudaFree(d_dst);
            return;
        }

        cv::Mat resized_emoji(box.height, box.width, CV_8UC4);
        cudaMemcpy(resized_emoji.data, d_dst, dstStep * box.height, cudaMemcpyDeviceToHost);

        if (box.x >= 0 && box.y >= 0 && (box.x + box.width <= image.cols) && (box.y + box.height <= image.rows))
        {
            cv::Mat roi = image(box);

            for (int y = 0; y < resized_emoji.rows; ++y)
            {
                for (int x = 0; x < resized_emoji.cols; ++x)
                {
                    cv::Vec4b &emoji_pixel = resized_emoji.at<cv::Vec4b>(y, x);
                    if (emoji_pixel[3] > 0)
                    {
                        cv::Vec3b &roi_pixel = roi.at<cv::Vec3b>(y, x);
                        roi_pixel = cv::Vec3b(
                            static_cast<uchar>(roi_pixel[0] * (1 - blendRatio) + emoji_pixel[0] * blendRatio),
                            static_cast<uchar>(roi_pixel[1] * (1 - blendRatio) + emoji_pixel[1] * blendRatio),
                            static_cast<uchar>(roi_pixel[2] * (1 - blendRatio) + emoji_pixel[2] * blendRatio));
                    }
                }
            }
        }

        cudaFree(d_src);
        cudaFree(d_dst);
    }
}

// Post-process the results and return scaled bounding boxes
std::vector<cv::Rect> postProcessAndReturnBoxes(cv::Mat &image, const std::vector<float> &output, int imgWidth, int imgHeight, float conf_thres, float iou_thres, std::vector<int> &classIds)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    // Perform NMS after processing output data
    std::vector<int> nms_indices = non_max_suppression_face(output, boxes, confidences, classIds, conf_thres, iou_thres, 10);

    // Calculate scale factor used during preprocessing
    float scale = std::min(float(imgWidth) / image.cols, float(imgHeight) / image.rows);

    // Scale the bounding boxes to match the original image size
    std::vector<cv::Rect> scaled_boxes;
    for (int idx : nms_indices)
    {
        cv::Rect scaled_box = scaleBoundingBox(boxes[idx], image.cols, image.rows, imgWidth, imgHeight, scale);
        scaled_boxes.push_back(scaled_box);
    }

    return scaled_boxes;
}
