#ifndef FACE_SWAP_H
#define FACE_SWAP_H

#include <string>
#include <vector>
#include <NvInfer.h>

// Bounding box struct for detected faces
struct BoundingBox
{
    int x, y, width, height;
};

// TensorRT engine loading function
nvinfer1::ICudaEngine *load_engine(const std::string &engine_file, nvinfer1::IRuntime *&runtime);

// CUDA image loading and processing functions
unsigned char *load_image_to_gpu(const std::string &file_path, int &width, int &height);
unsigned char *load_kitty_to_gpu(const std::string &file_path, int &width, int &height);
bool write_image_from_gpu(const std::string &file_path, unsigned char *d_image, int width, int height);

// TensorRT inference function
std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine *engine, void *input_data, int input_size);

// Face swapping function
void replace_with_kitty(unsigned char *img, int width, int height, const BoundingBox &face, unsigned char *kitty_emoji, int kitty_width, int kitty_height);

#endif
