# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4` -I/usr/local/include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -Isrc -Iinclude

# Libraries
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -L/usr/local/cuda/lib64 -L/usr/local/lib -lonnxruntime -lcudart -lnppicc -lnppig -lnppial -lnppidei -lnppist -lgtest -lgtest_main -lpthread -lstdc++

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

# Source files
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/nms.cpp \
       $(SRC_DIR)/preprocess.cpp \
       $(SRC_DIR)/inference.cpp \
       $(SRC_DIR)/postprocess.cpp \
       $(SRC_DIR)/util.cpp

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Test source files
TEST_SRCS = $(TEST_DIR)/test_nms.cpp \
            $(TEST_DIR)/test_preprocess.cpp \
            $(TEST_DIR)/test_util.cpp

# Test object files
TEST_OBJS = $(TEST_SRCS:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Target executable for main application
TARGET = $(BIN_DIR)/cudacatify

# Test executable
TEST_TARGET = $(BIN_DIR)/test_cudacatify

# Default target
all: $(TARGET)

# Ensure output directories exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Linking the final executable for main application
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Linking the test executable
$(TEST_TARGET): $(TEST_OBJS) $(filter-out $(OBJ_DIR)/main.o, $(OBJS)) | $(BIN_DIR)
	$(CXX) -DUNIT_TEST $(TEST_OBJS) $(filter-out $(OBJ_DIR)/main.o, $(OBJS)) -o $(TEST_TARGET) $(LDFLAGS)

# Rule to compile each source file into an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Rule to compile each test file into an object file
$(OBJ_DIR)/test_%.o: $(TEST_DIR)/test_%.cpp | $(OBJ_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Clean build artifacts
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET) $(TEST_TARGET)

# Run unit tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)
	rm -f $(TEST_TARGET)
