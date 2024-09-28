# Compiler settings
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# Compiler flags
CXXFLAGS = -O2 -std=c++11
NVCCFLAGS = -arch=sm_50 -O2 --use_fast_math

# Directories
INCLUDES = -I/usr/local/cuda/include -I./include -I/home/amd/Projects/CudaCatify/workspace/opencv-4.9.0/include

LIBS = -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu/ -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

# Files and target
TARGET = cuda-cat-face-swap
SRCS = src/main.cpp src/utils.cpp
CU_SRCS = src/face_swap.cu

# Rules
all: $(TARGET)

# Rule to compile CUDA and C++ source files
$(TARGET): $(SRCS) $(CU_SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRCS) $(CU_SRCS) -o $(TARGET) $(LIBS)

# Clean up object files and the executable
clean:
	rm -f $(TARGET) *.o
