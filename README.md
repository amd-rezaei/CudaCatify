
# CudaCatify

CudaCatify is a CUDA-accelerated application that performs object detection using YOLOv5 and replaces detected objects in an image with an emoji. The project leverages OpenCV for image processing, ONNX Runtime for YOLOv5 inference, and NVIDIA's NPP (NVIDIA Performance Primitives) for CUDA-based image manipulation, such as resizing and blending.

## Features
- Object detection with YOLOv5 using ONNX Runtime.
- Image preprocessing (resizing and padding) with NPP and OpenCV.
- Replace detected bounding boxes with an emoji image using NPP for resizing and blending.
- Supports blending between the original image and the emoji.
- Fully CUDA-accelerated for better performance on NVIDIA GPUs.

## Requirements

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (for NPP and GPU acceleration)
- [OpenCV](https://opencv.org/) (for image processing)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (for running YOLOv5 ONNX models)
- A CUDA-capable NVIDIA GPU

## Installation

### 1. Install CUDA Toolkit
Follow the official NVIDIA guide to install the CUDA Toolkit: [CUDA Toolkit Installation](https://developer.nvidia.com/cuda-downloads)

### 2. Install OpenCV
You can install OpenCV using a package manager (such as `apt` on Ubuntu) or by building it from source:

```bash
sudo apt update
sudo apt install libopencv-dev
```

### 3. Install ONNX Runtime
To install ONNX Runtime, you can follow the guide here: [ONNX Runtime Installation](https://onnxruntime.ai/docs/build/)

Or you can install it using `pip`:

```bash
pip install onnxruntime
```

### 4. Clone the Repository

```bash
git clone https://github.com/your-username/cudacatify.git
cd cudacatify
```

### 5. Build the Project

Make sure CUDA and OpenCV paths are correctly set up in your environment. Then, build the project using the following commands:

```bash
make
```

This will compile the code and generate the executable in the `bin/` directory.

## Usage

The program takes several command-line arguments:

```bash
./bin/cudacatify <yolov5_model.onnx> <input_image> <confidence_threshold> <iou_threshold> <emoji_image> <blend_ratio>
```

### Example:

```bash
./bin/cudacatify yolov5s.onnx input.jpg 0.5 0.5 emoji.png 0.5
```

### Arguments:

- `yolov5_model.onnx`: Path to the YOLOv5 ONNX model file.
- `input_image`: Path to the input image on which you want to perform detection.
- `confidence_threshold`: The confidence threshold for filtering YOLOv5 detections (0.0 to 1.0).
- `iou_threshold`: The IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).
- `emoji_image`: Path to the emoji image to replace detected objects with.
- `blend_ratio`: The blending ratio between the original image and the emoji (0.0 to 1.0).

### Output:

The program saves the resulting image with emojis in the current directory as `output_with_emojis.jpg`.

### Notes:
- `blend_ratio = 0.0`: Only the original image is kept.
- `blend_ratio = 1.0`: Only the emoji is visible in the detected areas.
- Values between `0.0` and `1.0` blend the emoji and the original image in the detected areas.

## Code Structure

- `src/main.cpp`: The main file that handles the application logic, including argument parsing, image loading, inference, and post-processing.
- `replaceWithEmojiInPostProcessNPP`: This function replaces detected bounding boxes with emojis, using NPP for CUDA-accelerated resizing and blending.

## Performance

The application leverages CUDA's NPP library for high-performance image manipulation, which significantly speeds up the image preprocessing and emoji replacement steps, especially for larger images or a high number of detections.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- YOLOv5: [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- ONNX Runtime: [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- OpenCV: [OpenCV GitHub](https://github.com/opencv/opencv)
- CUDA and NPP: [NVIDIA Developer Zone](https://developer.nvidia.com/)
