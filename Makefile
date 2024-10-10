# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I/home/amd/libs/onnxruntime/include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/home/amd/Projects/CudaCatify/include `pkg-config --cflags opencv4`

# Libraries
LDFLAGS = -L/home/amd/libs/onnxruntime/lib -lonnxruntime -lcudart -lnppicc -lnppig -lnppial -lnppidei -lnppist `pkg-config --libs opencv4`

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/nms.cpp \
       $(SRC_DIR)/preprocess.cpp \
       $(SRC_DIR)/inference.cpp \
       $(SRC_DIR)/postprocess.cpp \
       $(SRC_DIR)/util.cpp

# Object files
OBJS = $(OBJ_DIR)/main.o \
       $(OBJ_DIR)/nms.o \
       $(OBJ_DIR)/preprocess.o \
       $(OBJ_DIR)/inference.o \
       $(OBJ_DIR)/postprocess.o \
       $(OBJ_DIR)/util.o

# Target executable
TARGET = $(BIN_DIR)/cudacatify

# Default target
all: $(TARGET)

# Ensure output directories exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Linking the final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Rule to compile each source file into an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Clean build artifacts
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)
