#include "face_swap.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <iostream>
#include <stb_image.h>
#include <stb_image_write.h>

// Load image into GPU memory
unsigned char *load_image_to_gpu(const std::string &file_path, int &width, int &height)
{
    int channels;
    unsigned char *image = stbi_load(file_path.c_str(), &width, &height, &channels, STBI_rgb);
    if (!image)
    {
        std::cerr << "Failed to load image: " << file_path << std::endl;
        return nullptr;
    }

    int image_size = width * height * 3; // Assuming 3-channel RGB image

    unsigned char *d_image;
    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    stbi_image_free(image);
    return d_image;
}

// Load kitty emoji into GPU memory
unsigned char *load_kitty_to_gpu(const std::string &file_path, int &width, int &height)
{
    int channels;
    unsigned char *kitty_emoji = stbi_load(file_path.c_str(), &width, &height, &channels, STBI_rgb);
    if (!kitty_emoji)
    {
        std::cerr << "Failed to load kitty emoji: " << file_path << std::endl;
        return nullptr;
    }

    int emoji_size = width * height * 3; // Assuming 3-channel RGB emoji

    unsigned char *d_kitty_emoji;
    cudaMalloc(&d_kitty_emoji, emoji_size);
    cudaMemcpy(d_kitty_emoji, kitty_emoji, emoji_size, cudaMemcpyHostToDevice);

    stbi_image_free(kitty_emoji);
    return d_kitty_emoji;
}

// Write output image from GPU to disk
bool write_image_from_gpu(const std::string &file_path, unsigned char *d_image, int width, int height)
{
    int image_size = width * height * 3;
    unsigned char *h_image = (unsigned char *)malloc(image_size);
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    if (!stbi_write_jpg(file_path.c_str(), width, height, 3, h_image, 100))
    {
        std::cerr << "Failed to save output image: " << file_path << std::endl;
        free(h_image);
        return false;
    }

    free(h_image);
    return true;
}

// Replace face with kitty emoji
__global__ void replace_with_kitty_kernel(unsigned char *img, int img_width, int img_height,
                                          const BoundingBox face, unsigned char *kitty_emoji, int kitty_width, int kitty_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= face.x && x < face.x + face.width && y >= face.y && y < face.y + face.height)
    {
        int img_idx = (y * img_width + x) * 3;
        int emoji_x = (x - face.x) * kitty_width / face.width;
        int emoji_y = (y - face.y) * kitty_height / face.height;
        int emoji_idx = (emoji_y * kitty_width + emoji_x) * 3;

        img[img_idx] = kitty_emoji[emoji_idx];
        img[img_idx + 1] = kitty_emoji[emoji_idx + 1];
        img[img_idx + 2] = kitty_emoji[emoji_idx + 2];
    }
}

void replace_with_kitty(unsigned char *img, int width, int height, const BoundingBox &face, unsigned char *kitty_emoji, int kitty_width, int kitty_height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    replace_with_kitty_kernel<<<gridSize, blockSize>>>(img, width, height, face, kitty_emoji, kitty_width, kitty_height);
}
