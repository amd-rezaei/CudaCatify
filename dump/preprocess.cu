#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// CUDA kernel to preprocess image
__global__ void preprocess_kernel(float *input, unsigned char *image, int width, int height, int channels, int input_width, int input_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height)
        return;

    // Rescale image index
    int c = idx / (width * height);
    int h = (idx % (width * height)) / width;
    int w = idx % width;

    // Compute scaled coordinates
    float scale_w = float(input_width) / float(width);
    float scale_h = float(input_height) / float(height);
    int scaled_w = w * scale_w;
    int scaled_h = h * scale_h;

    // Normalize pixel values
    for (int i = 0; i < channels; i++)
    {
        input[c * input_width * input_height + scaled_h * input_width + scaled_w] = float(image[c * width * height + h * width + w]) / 255.0f;
    }
}

extern "C" void preprocess(float *d_input, unsigned char *image, int width, int height, int channels, int input_width, int input_height)
{
    // Call CUDA kernel with 1024 threads per block
    int threads_per_block = 1024;
    int blocks = (width * height + threads_per_block - 1) / threads_per_block;
    preprocess_kernel<<<blocks, threads_per_block>>>(d_input, image, width, height, channels, input_width, input_height);
}
