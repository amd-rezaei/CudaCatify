#ifndef FACE_SWAP_H
#define FACE_SWAP_H

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nppi.h>
#include <vector>
#include <string>

struct BoundingBox
{
    int x, y, width, height;
};

// Logger for TensorRT
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
};

// TensorRT Inference functions
nvinfer1::ICudaEngine *load_engine(const std::string &engine_file);
void allocate_buffers(nvinfer1::ICudaEngine *engine, void **buffers, int input_size, int output_size);
std::vector<BoundingBox> run_inference(nvinfer1::ICudaEngine *engine, void *input_data, int input_size);

// Utility functions for image handling
unsigned char *load_image_to_gpu(const std::string &file_path);
unsigned char *load_kitty_to_gpu(const std::string &file_path);
void write_image_from_gpu(const std::string &file_path, unsigned char *d_image);

// Face swap function
void replace_with_kitty(unsigned char *img, int width, int height, const BoundingBox &face, unsigned char *kitty_emoji, int kitty_width, int kitty_height);

#endif
