import cv2
import numpy as np
import onnxruntime as ort

# Single class label ("Face")
labels = ["Face"]

# Helper function to apply sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Helper function to preprocess the image and convert to Float16
def preprocess_image(image, input_width, input_height):
    original_height, original_width = image.shape[:2]

    # Resize the image while maintaining the aspect ratio
    scale = min(input_width / original_width, input_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create a padded image
    padded_image = np.full((input_height, input_width, 3), 128, dtype=np.float32)

    # Paste the resized image into the padded image
    x_offset = (input_width - new_width) // 2
    y_offset = (input_height - new_height) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_image

    # Normalize and convert to CHW format
    padded_image /= 255.0
    padded_image = np.transpose(padded_image, (2, 0, 1))  # HWC to CHW

    # Convert to Float16
    padded_image = padded_image.astype(np.float16)
    
    return np.expand_dims(padded_image, axis=0)  # Add batch dimension

# Function to clamp values before applying exp() to prevent overflow
def clamp(value, min_value, max_value):
    return np.maximum(min_value, np.minimum(value, max_value))

# Helper function to post-process the model's output and apply NMS
def post_process(output, conf_threshold, nms_threshold, img_width, img_height):
    boxes = []
    confidences = []
    class_ids = []

    num_boxes = output.shape[0]

    for i in range(num_boxes):
        # Get the bounding box and confidence values
        x_center, y_center, width, height, confidence, class_score = output[i][:6]

        confidence = sigmoid(confidence)  # Objectness score
        if confidence >= conf_threshold:
            # Convert the bounding box from center x, center y, width, height to x_min, y_min, width, height
            x_center = sigmoid(x_center) * img_width
            y_center = sigmoid(y_center) * img_height

            # Clamp values before applying exp() to avoid overflow
            width = clamp(np.exp(width) * img_width, 0, img_width)
            height = clamp(np.exp(height) * img_height, 0, img_height)

            # Convert to x_min, y_min, width, height
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)

            boxes.append([x_min, y_min, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(0)  # Assuming single class "Face"

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Return detected bounding boxes and confidence scores
    return [(boxes[i[0]], confidences[i[0]], class_ids[i[0]]) for i in indices]

# Draw bounding boxes and save the image
def draw_boxes(image, detections):
    for (box, confidence, class_id) in detections:
        x_min, y_min, width, height = box
        label = f"{labels[class_id]}: {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

        # Draw label
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Save output image
    cv2.imwrite("output_with_bboxes_python.jpg", image)
    print("Saved output image with bounding boxes as 'output_with_bboxes_python.jpg'")

# Run inference
def run_inference(model_path, image_path):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    input_shape = session.get_inputs()[0].shape
    preprocessed_image = preprocess_image(image, input_shape[3], input_shape[2])

    # Run inference
    output = session.run([output_name], {input_name: preprocessed_image})[0]

    # Reshape the output to have 6 values per detection (x_center, y_center, width, height, confidence, class_score)
    output = np.reshape(output, (-1, 6))  # 6 because there are 6 elements (float16)

    # Post-process output to filter out bounding boxes and apply NMS
    conf_threshold = 0.5
    nms_threshold = 0.4
    detections = post_process(output, conf_threshold, nms_threshold, img_width=image.shape[1], img_height=image.shape[0])

    # Draw and save the output image with bounding boxes
    draw_boxes(image, detections)

# Run the script
if __name__ == "__main__":
    run_inference('face_detection_yolov5s.onnx', 'faces.jpg')
