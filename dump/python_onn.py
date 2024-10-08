import onnxruntime as ort
import random
import numpy as np
import cv2
import time
import csv

class ort_v5:
    def __init__(self, img_path, onnx_model, conf_thres, iou_thres, img_size, classes):
        self.img_path = img_path
        self.onnx_model = onnx_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.names = classes
        self.ort_session = None

    def __call__(self):
        # Check if CUDAExecutionProvider is available, else fallback to CPU
        providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

        self.ort_session = ort.InferenceSession(self.onnx_model, providers=providers)

        # Load and process the image
        image_or = cv2.imread(self.img_path)
        output_image = self.detect_img(image_or)
        cv2.imwrite('./output_with_bboxes.jpg', output_image)  # Save the output image with bounding boxes
        print("Output saved as 'output_with_bboxes.jpg'")

    def detect_img(self, image_or):
        # Image preprocessing for the YOLOv5 model
        image, ratio, dwdh = self.letterbox(image_or, auto=False)
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # YOLOv5 expects a [1, 3, 640, 640] input (assuming the model was trained on 640x640 images)
        session = self.ort_session
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]: im}

        # ONNXRuntime inference
        t1 = time.time()
        outputs = session.run(outname, inp)
        t2 = time.time()
        print(f'ONNXRuntime Inference Time: {t2 - t1}s')

        # Log raw 25,200 x 16 values to a CSV file
        self.save_raw_outputs_to_csv(outputs[0])

        # Process the output tensor
        output = np.array(outputs[0])

        # Perform NMS (non-max suppression)
        out = self.non_max_suppression_face(output, self.conf_thres, self.iou_thres)[0]

        # Log NMS results to a CSV file
        self.save_nms_outputs_to_csv(out)

        # Draw results on the image
        result_img = self.result(image_or, ratio, dwdh, out)
        return result_img

    def save_raw_outputs_to_csv(self, raw_outputs):
        # Reshape raw_outputs to 25,200 x 16 format
        raw_outputs = raw_outputs.reshape(-1, 16)

        # Write to CSV
        with open('log_py.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            header = ['x_center', 'y_center', 'width', 'height', 'confidence'] + [f'class_prob_{i}' for i in range(10)] + ['extra']
            writer.writerow(header)

            # Write each row of raw detection values
            for row in raw_outputs:
                writer.writerow(row)

        print(f"Raw output data has been written to 'log_py.csv'.")

    def save_nms_outputs_to_csv(self, nms_outputs):
        # Write NMS results to CSV
        with open('nms_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            header = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class']
            writer.writerow(header)

            # Write each row of NMS filtered bounding boxes and their associated information
            for row in nms_outputs:
                x_min, y_min, x_max, y_max, score, class_id = row[:6]
                writer.writerow([x_min, y_min, x_max, y_max, score, int(class_id)])

        print(f"NMS results have been written to 'nms_results.csv'.")

    # Display results
    def result(self, img, ratio, dwdh, out):
        # Load class names from the provided class file
        names = self.class_name()
        colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

        for i, (x0, y0, x1, y1, score) in enumerate(out[:, 0:5]):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()

            score = round(float(score), 3)
            name = names[0]  # Assign a class name, modify as needed
            color = colors[name]
            name += f' {score}'

            # Draw the bounding box and label on the image
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return img

    def non_max_suppression_face(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        nc = prediction.shape[2] - 15  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after
        redundant = True
        multi_label = nc > 1  # multiple labels per box
        merge = False  # use merge-NMS

        t = time.time()
        output = [np.zeros((0, 16))] * prediction.shape[0]
        
        # Open a CSV file to write the values after the operation
        with open('after_conf_cls_multiplication.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['x_center', 'y_center', 'width', 'height', 'confidence'] + [f'class_prob_{i}' for i in range(10)]
            writer.writerow(header)

            for xi, x in enumerate(prediction):
                x = x[xc[xi]]  # filter out by confidence threshold
                
                if not x.shape[0]:
                    continue

                # Multiply class confidence with object confidence
                x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Write values to the CSV after this operation
                for row in x:
                    row_to_write = list(row[:16])  # Take the 16 values
                    writer.writerow(row_to_write)

                # Perform the rest of the processing as usual
                box = self.xywh2xyxy(x[:, :4])

                if multi_label:
                    i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
                    x = np.concatenate((box[i], x[i, j + 15, None], x[i, 5:15], j[:, None].astype(np.float32)), 1)
                else:
                    conf = np.max(x[:, 15:], axis=1, keepdims=True)
                    j = np.argmax(x[:, 15:], axis=1)
                    x = np.concatenate((box, conf, x[:, 5:15], j[:, None].astype(np.float32)), 1)
                    x = x[conf.flatten() > conf_thres]

                output[xi] = x
                if (time.time() - t) > time_limit:
                    break

        return output
    def xywh2xyxy(self, x):
        y = np.zeros_like(x) if isinstance(x, np.ndarray) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def class_name(self):
        classes = []
        with open(self.names, 'r') as file:
            while True:
                name = file.readline().strip('\n')
                if name == '':
                    break
                classes.append(name)
        return classes

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.img_size
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)


# Example usage
image = './sdfaces.jpg'  # Path to input image
weights = './yolov5m-face.onnx'  # YOLOv5 ONNX model
conf = 0.7
iou_thres = 0.5
img_size = 640
classes_txt = './classes.txt'  # Path to the class names file

ORT = ort_v5(image, weights, conf, iou_thres, (img_size, img_size), classes=classes_txt)
ORT()  # Run inference
