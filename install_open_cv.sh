#!/bin/bash

# Exit on error
set -e

# Update package lists and install dependencies
echo "Installing dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
                    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
                    libatlas-base-dev gfortran \
                    python3-dev python3-numpy python3-pip \
                    libxvidcore-dev libx264-dev libopenblas-dev liblapack-dev \
                    libeigen3-dev libhdf5-dev libhdf5-serial-dev \
                    libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
                    pkg-config

# Define OpenCV and OpenCV Contrib versions
OPENCV_VERSION="4.8.0"  # Set to at least 4.8.0 for better CUDA 12.x and Ada Lovelace support

# Create a directory for the installation
INSTALL_DIR=~/opencv_cuda_install
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Clone OpenCV and OpenCV Contrib repositories
echo "Cloning OpenCV and OpenCV Contrib repositories..."
git clone --branch $OPENCV_VERSION https://github.com/opencv/opencv.git
git clone --branch $OPENCV_VERSION https://github.com/opencv/opencv_contrib.git

# Create a build directory
mkdir -p opencv/build
cd opencv/build

# Configure the build with CMake, including CUDA and contrib modules
echo "Configuring the OpenCV build with CMake..."
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.9 \  # RTX 4070 CUDA compute capability
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_cudacodec=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON_EXECUTABLE=$(which python3) ..

# Build OpenCV using all available CPU cores
echo "Building OpenCV with CUDA support. This may take a while..."
make -j$(nproc)

# Install the compiled OpenCV libraries
echo "Installing OpenCV..."
sudo make install
sudo ldconfig

# Check if OpenCV was installed with CUDA support
echo "Verifying OpenCV installation with CUDA support..."
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda

echo "OpenCV with CUDA support installed successfully!"
