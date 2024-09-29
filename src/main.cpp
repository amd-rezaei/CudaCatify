#include "face_swap.h"
#include <iostream>
#include <string>
#include <vector>

// Use the TensorRT namespace explicitly or define a namespace alias
using namespace nvinfer1;

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <input_image> <kitty_emoji> <output_image>\n";
        return 1;
    }

    // Parse arguments
    std::string engine_file = argv[1];  // Path to TensorRT engine file (e.g., yolov4-tiny.engine)
    std::string input_image = argv[2];  // Path to input image (e.g., images/input_image.jpg)
    std::string kitty_emoji = argv[3];  // Path to kitty emoji (e.g., images/kitty_emoji.png)
    std::string output_image = argv[4]; // Path to output image (e.g., images/output_image.jpg)

    // Load TensorRT engine
    ICudaEngine *engine = load_engine(engine_file);
    if (!engine)
    {
        std::cerr << "Failed to load engine: " << engine_file << std::endl;
        return 1;
    }

    // Load input image into GPU
    unsigned char *d_image = load_image_to_gpu(input_image);
    if (!d_image)
    {
        std::cerr << "Failed to load input image: " << input_image << std::endl;
        return 1;
    }

    // Run YOLO object detection
    std::vector<BoundingBox> faces = run_inference(engine, d_image, 640 * 640 * 3);

    // Load kitty emoji into GPU
    unsigned char *d_kitty_emoji = load_kitty_to_gpu(kitty_emoji);
    if (!d_kitty_emoji)
    {
        std::cerr << "Failed to load kitty emoji: " << kitty_emoji << std::endl;
        cudaFree(d_image);
        return 1;
    }

    // Replace each detected face with the kitty emoji
    for (const auto &face : faces)
    {
        replace_with_kitty(d_image, 640, 640, face, d_kitty_emoji, 64, 64); // Assuming 64x64 kitty emoji
    }

    // Save the modified image back to disk
    write_image_from_gpu(output_image, d_image);

    // Free resources
    cudaFree(d_image);
    cudaFree(d_kitty_emoji);

    return 0;
}
