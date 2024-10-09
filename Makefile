# Libraries
LDFLAGS = -L/home/amd/libs/onnxruntime/lib -lonnxruntime -lcudart -lnppicc -lnppig -lnppial -lnppidei -lnppist `pkg-config --libs opencv4`

# Compiler and flags
CXX = g++
CXXFLAGS = -I/home/amd/libs/onnxruntime/include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc `pkg-config --cflags opencv4`

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Targets
TARGET = $(BIN_DIR)/cudacatify
OBJS = $(OBJ_DIR)/main.o

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

# Compile C++ source file (main.cpp is in src directory)
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
	$(CXX) -c $(SRC_DIR)/main.cpp -o $(OBJ_DIR)/main.o $(CXXFLAGS)

# Clean build artifacts
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/cudacatify
