# Compiler settings
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# Compiler flags
CXXFLAGS = -O2 -std=c++11
NVCCFLAGS = -arch=sm_50 -O2 --use_fast_math

# Directories
INCLUDES = -I/usr/local/cuda/include -I./include -I/usr/include/opencv4


LIBS = -L/usr/local/cuda/lib64 -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

# Files and target
TARGET = cuda-cat-face-swap
SRCS = src/main.cpp src/face_swap.cu src/utils.cpp
OBJS = main.o face_swap.o utils.o

# Rules
all: $(TARGET)

# Rule to compile CUDA and C++ source files
$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LIBS)

# Clean up object files and the executable
clean:
	rm -f $(TARGET) $(OBJS)
