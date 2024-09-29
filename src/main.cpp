#include "face_swap.h"
#include <iostream>
#include <string>
#include <vector>

// Use the TensorRT namespace explicitly
using namespace nvinfer1;

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <input_image> <kitty_emoji> <output_image>\n";
        return 1;
    }

    // Parse arguments
    std::string engine_file = argv[1];
    std::string input_image = argv[2];
    std::string kitty_emoji = argv[3];
    std::string output_image = argv[4];

    // Load TensorRT engine
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = load_engine(engine_file, runtime);
    if (!engine)
    {
        std::cerr << "Failed to load engine: " << engine_file << std::endl;
        return 1;
    }

    // Load input image into GPU
    int img_width, img_height;
    unsigned char *d_image = load_image_to_gpu(input_image, img_width, img_height);
    if (!d_image)
    {
        std::cerr << "Failed to load input image: " << input_image << std::endl;
        return 1;
    }

    // Run YOLO object detection using yolov4-tiny.engine
    std::vector<BoundingBox> faces = run_inference(engine, d_image, img_width * img_height * 3);

    // Load kitty emoji into GPU
    int kitty_width, kitty_height;
    unsigned char *d_kitty_emoji = load_kitty_to_gpu(kitty_emoji, kitty_width, kitty_height);
    if (!d_kitty_emoji)
    {
        std::cerr << "Failed to load kitty emoji: " << kitty_emoji << std::endl;
        cudaFree(d_image);
        return 1;
    }

    // Replace each detected face with the kitty emoji
    for (const auto &face : faces)
    {
        replace_with_kitty(d_image, img_width, img_height, face, d_kitty_emoji, kitty_width, kitty_height);
    }

    // Save the modified image back to disk
    if (!write_image_from_gpu(output_image, d_image, img_width, img_height))
    {
        std::cerr << "Failed to write output image: " << output_image << std::endl;
    }

    // Free resources
    cudaFree(d_image);
    cudaFree(d_kitty_emoji);
    // runtime->destroy();

    return 0;
}
