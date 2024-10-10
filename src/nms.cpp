// nms.cpp
#include "nms.hpp"

// Perform Non-Maximum Suppression (NMS)
std::vector<int> performNMS(std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &class_ids, float iou_thres)
{
    std::vector<int> indices;
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0); // Fill indices with 0, 1, ..., n-1

    // Sort by confidence scores
    std::sort(idxs.begin(), idxs.end(), [&confidences](int i1, int i2)
              { return confidences[i1] > confidences[i2]; });

    while (!idxs.empty())
    {
        int idx = idxs[0];
        indices.push_back(idx);
        std::vector<int> newIdxs;

        for (size_t i = 1; i < idxs.size(); ++i)
        {
            int currentIdx = idxs[i];

            // Calculate the IoU (Intersection over Union)
            float interArea = (boxes[idx] & boxes[currentIdx]).area();
            float unionArea = boxes[idx].area() + boxes[currentIdx].area() - interArea;
            float iou = interArea / unionArea;

            // Keep boxes that don't overlap too much (IoU < threshold)
            if (iou <= iou_thres)
            {
                newIdxs.push_back(currentIdx);
            }
        }

        idxs = std::move(newIdxs);
    }

    return indices;
}

// Non-Maximum Suppression (NMS) for YOLOv5 detection output
std::vector<int> non_max_suppression_face(const std::vector<float> &output, std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &class_ids, float conf_thres, float iou_thres, int num_classes)
{
    int numBoxes = output.size() / 16; // Assuming each box has 16 values (x, y, w, h, obj_conf, class_probs, etc.)

    for (int i = 0; i < numBoxes; ++i)
    {
        int index = i * 16;

        // Object confidence
        float confidence = output[index + 4];

        if (confidence >= conf_thres)
        {
            // Extract box coordinates and dimensions (center x, center y, width, height)
            float x_center = output[index];
            float y_center = output[index + 1];
            float width = output[index + 2];
            float height = output[index + 3];

            // Store box coordinates (in pixels)
            boxes.push_back(cv::Rect(static_cast<int>(x_center - width / 2), static_cast<int>(y_center - height / 2), static_cast<int>(width), static_cast<int>(height)));

            // Store confidence score
            confidences.push_back(confidence);

            // Determine best class
            float max_class_score = -1.0f;
            int best_class = -1;
            for (int j = 0; j < num_classes; ++j)
            {
                float class_prob = output[index + 5 + j] * confidence;
                if (class_prob > max_class_score)
                {
                    best_class = j;
                    max_class_score = class_prob;
                }
            }
            class_ids.push_back(best_class);
        }
    }

    // Apply Non-Maximum Suppression (NMS)
    return performNMS(boxes, confidences, class_ids, iou_thres);
}
