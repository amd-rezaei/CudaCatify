#include "face_swap.h"
#include <iostream>

// Use the TensorRT namespace explicitly or define a namespace alias
using namespace nvinfer1;

int main()
{
    // Load TensorRT engine
    ICudaEngine *engine = load_engine("yolov4-tiny.engine");

    // Load input image into GPU
    unsigned char *d_image = load_image_to_gpu("images/input_image.jpg");

    // Run YOLO object detection
    std::vector<BoundingBox> faces = run_inference(engine, d_image, 640 * 640 * 3);

    // Load kitty emoji into GPU
    unsigned char *d_kitty_emoji = load_kitty_to_gpu("images/kitty_emoji.png");

    // Replace each detected face with the kitty emoji
    for (const auto &face : faces)
    {
        replace_with_kitty(d_image, 640, 640, face, d_kitty_emoji, 64, 64); // Assuming 64x64 kitty emoji
    }

    // Save the modified image back to disk
    write_image_from_gpu("images/output_image.jpg", d_image);

    // Free resources
    cudaFree(d_image);
    cudaFree(d_kitty_emoji);

    // No need to explicitly release engine, just let it go out of scope

    return 0;
}
