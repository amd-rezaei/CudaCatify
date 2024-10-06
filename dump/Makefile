CXX = g++
NVCC = nvcc
CXXFLAGS = -I/home/amd/libs/onnxruntime/include -I/usr/local/cuda/include `pkg-config --cflags opencv4` -I../include

LDFLAGS = -L/home/amd/libs/onnxruntime/lib -lonnxruntime -lcudart `pkg-config --libs opencv4`

# Targets
TARGET = yolov4_inference
OBJS = main.o preprocess.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.cpp
	$(CXX) -c main.cpp $(CXXFLAGS)

preprocess.o: preprocess.cu
	$(NVCC) -c preprocess.cu -o preprocess.o -I/usr/include/opencv4 # Add OpenCV include path here

clean:
	rm -f $(OBJS) $(TARGET)
