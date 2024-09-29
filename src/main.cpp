#include "face_swap.h"
#include <iostream>
#include <string>
#include <vector>
#include <npp.h> // Include NPP header

// Use the TensorRT namespace explicitly
using namespace nvinfer1;

// Define constants for YOLO input size
const int YOLO_INPUT_WIDTH = 832;
const int YOLO_INPUT_HEIGHT = 832;

// NPP resize function
unsigned char *resize_image_to_416(unsigned char *d_src_image, int src_width, int src_height)
{
    // Allocate memory for the resized image on the GPU
    unsigned char *d_resized_image;
    int resized_image_size = YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * 3; // Assuming 3 channels (RGB)
    cudaMalloc(&d_resized_image, resized_image_size);

    // Set up NPP structures for source and destination images
    NppiSize src_size = {src_width, src_height};
    NppiRect src_roi = {0, 0, src_width, src_height};

    NppiSize dst_size = {YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT};
    NppiRect dst_roi = {0, 0, YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT};

    int src_step = src_width * 3 * sizeof(unsigned char);        // Input image step (3 channels)
    int dst_step = YOLO_INPUT_WIDTH * 3 * sizeof(unsigned char); // Output image step (3 channels)

    // Perform the resize operation
    NppStatus status = nppiResize_8u_C3R(d_src_image, src_step, src_size, src_roi,
                                         d_resized_image, dst_step, dst_size, dst_roi,
                                         NPPI_INTER_LINEAR); // Use linear interpolation for resizing

    // Check for NPP errors
    if (status != NPP_SUCCESS)
    {
        std::cerr << "NPP Resize failed with status: " << status << std::endl;
        cudaFree(d_resized_image);
        return nullptr;
    }

    return d_resized_image;
}

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
    printf("Image loaded with size: %d x %d\n", img_width, img_height);

    // Resize the image to 416x416 using NPP
    unsigned char *d_resized_image = resize_image_to_416(d_image, img_width, img_height);
    if (!d_resized_image)
    {
        std::cerr << "Failed to resize input image" << std::endl;
        cudaFree(d_image);
        return 1;
    }
    printf("Image resized to %d x %d\n", YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT);

    // Run YOLO object detection using the resized image and correct size
    std::vector<BoundingBox> faces = run_inference(engine, d_resized_image, YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * 3);

    // Load kitty emoji into GPU
    int kitty_width, kitty_height;
    unsigned char *d_kitty_emoji = load_kitty_to_gpu(kitty_emoji, kitty_width, kitty_height);
    if (!d_kitty_emoji)
    {
        std::cerr << "Failed to load kitty emoji: " << kitty_emoji << std::endl;
        cudaFree(d_image);
        cudaFree(d_resized_image);
        return 1;
    }

    // Replace each detected face with the kitty emoji
    for (const auto &face : faces)
    {
        replace_with_kitty(d_resized_image, YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT, face, d_kitty_emoji, kitty_width, kitty_height);
    }

    // Save the modified image back to disk
    if (!write_image_from_gpu(output_image, d_resized_image, YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT))
    {
        std::cerr << "Failed to write output image: " << output_image << std::endl;
    }

    // Free resources
    cudaFree(d_image);
    cudaFree(d_resized_image);
    cudaFree(d_kitty_emoji);
    // runtime->destroy();

    return 0;
}
