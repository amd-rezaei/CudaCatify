# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O2 -g  # Enable debugging with -g
NVCCFLAGS = -g -G  # Enable device-side debugging with -G

# CUDA and TensorRT paths
CUDA_PATH ?= /usr/local/cuda-12.4
TENSORRT_PATH ?= /usr/lib/x86_64-linux-gnu

# Libraries for CUDA, TensorRT, and NPP (NVIDIA Performance Primitives)
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda
NPP_LIBS = -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -lnvjpeg
TENSORRT_LIBS = -L$(TENSORRT_PATH) -lnvinfer -lnvinfer_plugin -lnvonnxparser

# Include paths
INCLUDE_PATHS = -I$(CUDA_PATH)/include -I$(TENSORRT_PATH) -I./include

# Python environment and requirements
PYTHON = python3
REQUIREMENTS = py_scripts/requirements.txt
PY_SCRIPTS = py_scripts/yolo_engine.py

# Engine file created by the Python script
ENGINE_FILE = models/yolov4-tiny.engine

# Source files
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/face_swap.cu $(SRC_DIR)/utils.cpp $(SRC_DIR)/stb_image_implementation.cpp $(SRC_DIR)/utils_model.cu
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/face_swap.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/stb_image_implementation.o $(OBJ_DIR)/utils_model.o

# Output executable
TARGET = $(BIN_DIR)/face_swap

# Default target: Compile the project
all: $(ENGINE_FILE) $(TARGET)

# Ensure the Python environment is set up and the engine file is created
$(ENGINE_FILE): $(REQUIREMENTS) $(PY_SCRIPTS)
	@if [ ! -f $(ENGINE_FILE) ]; then \
		echo "TensorRT engine file not found, running conversion..."; \
		$(PYTHON) -m pip install -r $(REQUIREMENTS); \
		$(PYTHON) $(PY_SCRIPTS); \
	else \
		echo "TensorRT engine file found: $(ENGINE_FILE)"; \
	fi

# Create the directories if they don't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile the object files for C++ sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Compile the object files for CUDA sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Link the object files into the final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(OBJS) $(CUDA_LIBS) $(NPP_LIBS) $(TENSORRT_LIBS) -o $@

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
