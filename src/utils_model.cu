#include <cuda_runtime.h>

// CUDA kernel to convert unsigned char image data to float
__global__ void convert_image_to_float_kernel(unsigned char *input, float *output, int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels)
    {
        output[idx] = static_cast<float>(input[idx]) / 255.0f;
    }
}

// CUDA kernel to convert float image data back to unsigned char
__global__ void convert_float_to_uchar_kernel(float *input, unsigned char *output, int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels)
    {
        output[idx] = static_cast<unsigned char>(input[idx] * 255.0f);
    }
}

// Wrapper function for convert_image_to_float kernel
void convert_image_to_float(unsigned char *input, float *output, int num_pixels)
{
    int threads_per_block = 256;
    int blocks_per_grid = (num_pixels + threads_per_block - 1) / threads_per_block;
    convert_image_to_float_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, num_pixels);
}

// Wrapper function for convert_float_to_uchar kernel
void convert_float_to_uchar(float *input, unsigned char *output, int num_pixels)
{
    int threads_per_block = 256;
    int blocks_per_grid = (num_pixels + threads_per_block - 1) / threads_per_block;
    convert_float_to_uchar_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, num_pixels);
}
