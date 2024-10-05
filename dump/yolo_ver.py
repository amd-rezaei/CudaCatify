import onnxruntime as ort
import numpy as np
import cv2

# Load YOLO ONNX model
session = ort.InferenceSession("yolov4-tiny.onnx")

# Define labels
labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "TVmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Preprocess the input image
def preprocess(image, input_shape):
    ih, iw = input_shape
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    
    image_resized = cv2.resize(image, (nw, nh))
    image_padded = np.full((ih, iw, 3), 128)
    image_padded[(ih - nh) // 2: (ih - nh) // 2 + nh, (iw - nw) // 2: (iw - nw) // 2 + nw, :] = image_resized
    
    image = image_padded.astype(np.float32) / 255.0
    image = np.transpose(image, [2, 0, 1])  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Non-Maximum Suppression (NMS) to remove duplicate detections
def non_maximum_suppression(boxes, confidences, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

# Postprocess the outputs to extract bounding boxes and class labels
def postprocess(outputs, image_shape, confidence_threshold=0.5):
    # YOLOv4 ONNX model typically has two outputs
    boxes = outputs[0]  # Shape: (1, num_boxes, 4)
    scores = outputs[1]  # Shape: (1, num_boxes, num_classes)

    ih, iw, _ = image_shape
    detected_boxes = []
    detected_confidences = []
    detected_class_ids = []

    # Iterate over boxes and scores
    for i in range(boxes.shape[1]):  # Iterate over all boxes
        box = boxes[0, i]  # Each box is of shape (4,)
        class_scores = scores[0, i]  # Class scores for this box

        # Get the highest confidence and class index
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        # Apply confidence threshold
        if confidence > confidence_threshold:
            # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
            center_x, center_y, width, height = box[0]
            x1 = int((center_x - width / 2) * iw)
            y1 = int((center_y - height / 2) * ih)
            x2 = int((center_x + width / 2) * iw)
            y2 = int((center_y + height / 2) * ih)

            detected_boxes.append([x1, y1, x2, y2])
            detected_confidences.append(float(confidence))
            detected_class_ids.append(class_id)

    return detected_boxes, detected_confidences, detected_class_ids

# Draw bounding boxes on the image
def draw_boxes(image, boxes, confidences, class_ids, indices):
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Main function to run inference and save output
def run_inference(image_path, output_image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    input_shape = (416, 416)  # Expected input size for YOLO model
    preprocessed_image = preprocess(image, input_shape)

    # Run the model
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})

    # Postprocess the results
    boxes, confidences, class_ids = postprocess(outputs, image.shape)

    # Apply Non-Maximum Suppression to filter out duplicate boxes
    nms_indices = non_maximum_suppression(boxes, confidences, iou_threshold=0.4)

    # Print detected objects for debugging
    print("Detected objects after NMS:")
    for i in nms_indices:
        print(f"Object {i+1}: Class = {labels[class_ids[i]]}, Confidence = {confidences[i]:.2f}, Box = {boxes[i]}")

    # Draw bounding boxes and save the image
    draw_boxes(image, boxes, confidences, class_ids, nms_indices)
    cv2.imwrite(output_image_path, image)
    print(f"Saved output image with bounding boxes as '{output_image_path}'")

# Run inference on an example image
run_inference("outps.jpg", "output_with_bboxes.jpg")
