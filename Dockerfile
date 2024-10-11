# Use the Ubuntu 20.04 base image (you can switch this to a newer version if needed)
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies: development tools, OpenCV, CUDA, Google Test, and other necessary libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    wget \
    unzip \
    git \
    libopencv-dev \
    libgtest-dev \
    nvidia-cuda-toolkit \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/gtest

# Build and install Google Test (GTest)
RUN cmake . && make && cp /usr/src/googletest/googletest/lib/libgtest*.a /usr/lib

# Download and install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.14.1.tgz \
    && cp -r onnxruntime-linux-x64-1.14.1/include/* /usr/local/include/ \
    && cp onnxruntime-linux-x64-1.14.1/lib/libonnxruntime.so* /usr/local/lib/ \
    && ldconfig

# Create project directory
WORKDIR /usr/src/app

# Copy the source code into the container
COPY . .

# Run Make to build the project
RUN make

# Clean up build artifacts
RUN make clean

# Run unit tests
CMD ["make", "test"]
