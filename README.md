
# CudaCatify

CudaCatify is a CUDA-accelerated image and video processing application that performs face detection using a YOLOv5 model. It replaces detected faces with emoji overlays using OpenCV and NVIDIA's NPP library for efficient image manipulation on the GPU.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installing Dependencies](#installing-dependencies)
- [How to Get the YOLOv5 ONNX Model](#how-to-get-the-yolov5-onnx-model)
- [Project Structure](#project-structure)
- [How to Build](#how-to-build)
- [How to Run the Application](#how-to-run-the-application)
- [Running Unit Tests](#running-unit-tests)
- [Sample Output](#sample-output)
- [Cleaning the Build](#cleaning-the-build)
- [Credits](#credits)
- [License](#license)

## Features

- **Face detection** using a YOLOv5 model.
- **Emoji overlay** on detected faces.
- CUDA acceleration using **NVIDIA Performance Primitives (NPP)** for efficient image manipulation.
- **Unit tests** for core functionalities like NMS (Non-Maximum Suppression), preprocessing, and utility functions using **Google Test**.

## Prerequisites

Before building and running the project, ensure you have the following installed:

1. **CUDA Toolkit**: Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
2. **ONNX Runtime**: For running the YOLOv5 model.
3. **OpenCV**: Required for image and video processing.
4. **Google Test**: For running unit tests.

### Installing Dependencies

To install the necessary dependencies on Ubuntu, you can run the following commands:

```bash
# Update package list and install OpenCV, Google Test, and CMake
sudo apt-get update
sudo apt-get install -y libopencv-dev libgtest-dev cmake nvidia-cuda-toolkit

# Google Test setup
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp *.a /usr/lib
```

### ONNX Runtime Installation

To install ONNX Runtime, follow these steps:

```bash
# Download the ONNX Runtime tarball
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz

# Extract the tarball
tar -xzf onnxruntime-linux-x64-1.14.1.tgz

# Move headers and libraries to system directories
sudo cp -r onnxruntime-linux-x64-1.14.1/include/* /usr/local/include/
sudo cp onnxruntime-linux-x64-1.14.1/lib/libonnxruntime.so* /usr/local/lib/

# Update the shared library cache
sudo ldconfig

# Clean up the tarball and extracted files
rm -rf onnxruntime-linux-x64-1.14.1.tgz onnxruntime-linux-x64-1.14.1
```

## How to Get the YOLOv5 ONNX Model

This project includes a Git submodule for the YOLOv5 model, which contains the necessary ONNX model files. To access it, ensure that you have cloned the submodule as part of the project.

If you haven't initialized the submodule yet, run the following commands after cloning the repository:

```bash
git submodule init
git submodule update
```

The ONNX model can be found in the following location after initializing the submodule:

```bash
./submodules/FaceID-YOLOV5.ArcFace/yolov5m-face.onnx
```

When running the application, you can use this model by specifying its path:

```bash
./bin/cudacatify "./submodules/FaceID-YOLOV5.ArcFace/yolov5m-face.onnx" "/path/to/input/image.jpg / video/video.mp4" 
```

This will process the input image using the ONNX model from the submodule and apply the emoji overlay to detected faces.

## Project Structure

```bash
.
├── bin/                  # Compiled binaries
├── obj/                  # Compiled object files
├── src/                  # Source files
│   ├── main.cpp          # Main application logic
│   ├── nms.cpp           # Non-Maximum Suppression (NMS) logic
│   ├── preprocess.cpp    # Preprocessing (image resizing and formatting)
│   ├── inference.cpp     # Inference logic for ONNX model
│   ├── postprocess.cpp   # Postprocessing logic
│   └── util.cpp          # Utility functions (e.g., file handling, etc.)
├── tests/                # Unit test files
│   ├── test_nms.cpp      # Test cases for NMS
│   ├── test_preprocess.cpp # Test cases for preprocessing
│   └── test_util.cpp     # Test cases for utility functions
└── Makefile              # Makefile for building and running the project
```

## How to Build

To build the project, simply run:

```bash
make
```

This will compile the source files into object files and create the final executable `cudacatify` in the `bin/` directory.

## How to Run the Application

To run the main application, first build the project using:

```bash
make
```

This will compile the source files into object files and create the final executable `cudacatify` in the `bin/` directory.

To run the application, execute the following command:

```bash
./bin/cudacatify "/path/to/yolov5m-face.onnx" "/path/to/input/image.jpg"
```

Replace `/path/to/yolov5m-face.onnx` with the actual path to your YOLOv5 ONNX model, and `/path/to/input/image.jpg` with the path to the input image you want to process.

This will process the input image, detect faces, and replace them with the specified emoji.

## Running Unit Tests

Unit tests for the core components (such as NMS, preprocessing, and utility functions) can be run using the following command:

```bash
make test
```

This will compile and execute the test suite using Google Test. The results of the tests will be displayed in the terminal.

## Sample Output

Here are some examples of the output generated by the application. The original image (left) and the processed image with emoji overlays (right) are shown below:

| Original GIF | Processed GIF |
|--------------|---------------|
| ![Original](samples/pedestrian.gif) | ![Processed](samples/pedestrian_private.gif) |

## Cleaning the Build

To clean up all the compiled object files and binaries, run:

```bash
make clean
```

This will remove the object files in the `obj/` directory and the executable files in the `bin/` directory.

## Credits

- Open source videos and images from [Pexels](https://www.pexels.com).
- ONNX model used in this project is from the [FaceID--YOLOV5.ArcFace](https://github.com/PhucNDA/FaceID--YOLOV5.ArcFace.git) repository included in the submodules.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
