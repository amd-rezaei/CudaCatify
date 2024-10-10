
# CudaCatify

CudaCatify is a CUDA-accelerated image and video processing application that performs face detection using a YOLOv5 model. It replaces detected faces with emoji overlays using OpenCV and NVIDIA's NPP library for efficient image manipulation on the GPU.

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

You can install these dependencies on Ubuntu using the following commands:
```bash
# Install OpenCV
sudo apt-get install libopencv-dev

# Install Google Test
sudo apt-get install libgtest-dev cmake
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp *.a /usr/lib
```

You also need to ensure that ONNX Runtime is installed and accessible. You can download and build ONNX Runtime from its official [GitHub repository](https://github.com/microsoft/onnxruntime).

## How to Get the YOLOv5 ONNX Model

This project includes a Git submodule for the YOLOv5 model, which contains the necessary ONNX model files. To access it, ensure that you have cloned the submodule as part of the project.

If you haven't initialized the submodule yet, run the following commands after cloning the repository:

```bash
git submodule init
git submodule update
```

This will fetch the contents of the `FaceID-YOLOV5.ArcFace` submodule.

The ONNX model can be found in the following location after initializing the submodule:

```bash
./submodules/FaceID-YOLOV5.ArcFace/yolov5m-face.onnx
```

When running the application, you can use this model by specifying its path:

```bash
./bin/cudacatify "./submodules/FaceID-YOLOV5.ArcFace/yolov5m-face.onnx" "/path/to/input/image.jpg"
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

## Cleaning the Build

To clean up all the compiled object files and binaries, run:

```bash
make clean
```

This will remove the object files in the `obj/` directory and the executable files in the `bin/` directory.

## Additional Notes

- Ensure that the paths to ONNX Runtime, CUDA, and OpenCV libraries and include directories are correctly set in the `Makefile`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
