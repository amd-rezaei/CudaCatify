# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O2

# CUDA and TensorRT paths
CUDA_PATH ?= /usr/local/cuda-12.4
TENSORRT_PATH ?= /usr/lib/x86_64-linux-gnu

# Libraries for CUDA and TensorRT
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lnppc -lnppial -lnppicc -lnppidei -lnvjpeg
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

SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/face_swap.cu $(SRC_DIR)/utils.cpp
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/face_swap.o $(OBJ_DIR)/utils.o

# Output executable
TARGET = $(BIN_DIR)/face_swap

# Default target: Compile the project
all: $(ENGINE_FILE) $(TARGET)

# Ensure the Python environment is set up and the engine file is created
$(ENGINE_FILE): $(REQUIREMENTS) $(PY_SCRIPTS)
	@echo "Checking and installing Python packages..."
	$(PYTHON) -m pip install -r $(REQUIREMENTS)  # Install requirements if not already installed
	@echo "Running Python script to generate TensorRT engine..."
	$(PYTHON) $(PY_SCRIPTS)

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
	$(NVCC) $(INCLUDE_PATHS) -c $< -o $@

# Link the object files into the final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(OBJS) $(CUDA_LIBS) $(TENSORRT_LIBS) -o $@

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(ENGINE_FILE)
