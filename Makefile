# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -Isrc `pkg-config --cflags opencv4`

# Libraries
LDFLAGS = -L/usr/local/lib -lonnxruntime -lcudart -lnppicc -lnppig -lnppial -lnppidei -lnppist `pkg-config --libs opencv4` -lgtest -lgtest_main -lpthread

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests
SUBMODULE_DIR = submodules

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

# Test source files
TEST_SRCS = $(TEST_DIR)/test_nms.cpp \
            $(TEST_DIR)/test_preprocess.cpp \
            $(TEST_DIR)/test_util.cpp

# Test object files
TEST_OBJS = $(OBJ_DIR)/test_nms.o \
            $(OBJ_DIR)/test_preprocess.o \
            $(OBJ_DIR)/test_util.o

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

# Linking the test executable (define UNIT_TEST, and avoid compiling the main application logic)
$(TEST_TARGET): $(TEST_OBJS) $(filter-out $(OBJ_DIR)/main.o, $(OBJS)) | $(BIN_DIR)
	$(CXX) -DUNIT_TEST $(TEST_OBJS) $(filter-out $(OBJ_DIR)/main.o, $(OBJS)) -o $(TEST_TARGET) $(LDFLAGS) `pkg-config --libs opencv4` -lgtest -lgtest_main -lpthread

# Rule to compile each source file into an object file (for the main application)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Rule to compile each test file into an object file
$(OBJ_DIR)/test_%.o: $(TEST_DIR)/test_%.cpp | $(OBJ_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Clean build artifacts
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET) $(TEST_TARGET)

# Run unit tests (no external arguments are needed for unit tests)
test: $(TEST_TARGET)
	./$(TEST_TARGET)
	rm -f $(TEST_TARGET)
