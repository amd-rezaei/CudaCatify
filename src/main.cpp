// main.cpp
#ifdef UNIT_TEST
// This section will only compile for unit testing

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#else
// This section will compile for the main application logic
#include "util.hpp"
#include "inference.hpp"
#include "postprocess.hpp"
#include <cxxopts.hpp>
#include <opencv.hpp>
#include <filesystem>
#include <iostream>

int main(int argc, char *argv[])
{
    cxxopts::Options options("cudacatify", "Process video or image with emoji overlays");

    options.add_options()("emoji", "Path to the emoji image", cxxopts::value<std::string>()->default_value("./data/images/kitty_emoji.png"))("conf_thres", "Confidence threshold", cxxopts::value<float>()->default_value("0.5"))("iou_thres", "IoU threshold", cxxopts::value<float>()->default_value("0.5"))("blend_thresh", "Blend threshold", cxxopts::value<float>()->default_value("1.0"))("output_dir", "Output directory", cxxopts::value<std::string>()->default_value("./results/"))("onnx_model", "Path to the ONNX model", cxxopts::value<std::string>())("input", "Path to the input (image or video)", cxxopts::value<std::string>())("help", "Print help");

    options.parse_positional({"onnx_model", "input"});
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("onnx_model") || !result.count("input"))
    {
        std::cerr << "Error: <onnx_model> and <input (image or video)> are required." << std::endl;
        std::cout << options.help() << std::endl;
        return -1;
    }

    std::string onnx_model = result["onnx_model"].as<std::string>();
    std::string input = result["input"].as<std::string>();
    std::string optional_emoji = result["emoji"].as<std::string>();
    float optional_conf_thres = result["conf_thres"].as<float>();
    float optional_iou_thres = result["iou_thres"].as<float>();
    float optional_blend_thresh = result["blend_thresh"].as<float>();
    std::string optional_output_dir = result["output_dir"].as<std::string>();

    std::filesystem::create_directories(optional_output_dir);
    std::string extension = std::filesystem::path(input).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png")
    {
        cv::Mat image = cv::imread(input);
        if (image.empty())
        {
            std::cerr << "Error: Could not load image!" << std::endl;
            return -1;
        }

        std::vector<int> classIds;
        int imgWidth = image.cols;
        int imgHeight = image.rows;
        std::vector<cv::Rect> scaled_boxes = runInference(onnx_model, image, optional_conf_thres, optional_iou_thres, classIds, imgWidth, imgHeight);
        replaceWithEmojiInPostProcess(image, scaled_boxes, optional_emoji, optional_blend_thresh);

        std::string baseName = getBaseName(input);
        std::string output_file = optional_output_dir + "/" + baseName + "_private.jpg";
        cv::imwrite(output_file, image);
        std::cout << "Output image saved as: " << output_file << std::endl;
    }
    else if (extension == ".mp4" || extension == ".avi")
    {
        runInferenceVideo(onnx_model, input, optional_emoji, optional_conf_thres, optional_iou_thres, optional_blend_thresh, optional_output_dir);
    }
    else
    {
        std::cerr << "Error: Unsupported file format!" << std::endl;
        return -1;
    }

    return 0;
}
#endif