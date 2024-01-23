import os
import sys
import math
import cv2
import numpy as np
import subprocess
import onnxruntime

__labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def _read_image(img, target_size=320):
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width**2 + img_height**2) / (new_width**2 + new_height**2)
    )

    img = cv2.resize(img, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height

    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    img = cv2.resize(img, (target_size, target_size))

    image_data = img.astype("float32") / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    return image_data, resize_factor, pad_left, pad_top


def _postprocess(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top = int(round((y - h * 0.5 - pad_top) * resize_factor))
            width = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        detections.append(
            {"class": __labels[class_id], "score": float(score), "box": box}
        )

    return detections


class NudeDetector:
    def __init__(self, providers=None):
        model_path = os.path.join(os.path.dirname(__file__), "filter.onnx")
        # access read model
        if sys.platform == 'win32':
            username = os.environ.get('USERNAME') or os.environ.get('USER')
            model_cmd = f'icacls "{model_path}" /grant:r "{username}:(R,W)" /T'
            if os.environ.get('DEBUG', 'False') == 'True':
                # not silence run
                os.system(model_cmd)
            else:
                # silence run
                subprocess.run(model_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'linux':
            # access read model
            model_cmd = f"chmod +x {model_path}"
            if os.environ.get('DEBUG', 'False') == 'True':
                # not silence run
                os.system(model_cmd)
            else:
                # silence run
                subprocess.run(model_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # use cpu as with cuda on onnx can be problem
        if providers is None:
            providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers:
            provider = ["CUDAExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=provider,
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]  # 320
        self.input_height = input_shape[3]  # 320
        self.input_name = model_inputs[0].name

    def detect(self, img):
        preprocessed_image, resize_factor, pad_left, pad_top = _read_image(
            img, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs, resize_factor, pad_left, pad_top)

        return detections

    def censor(self, img, classes=None):
        detections = self.detect(img)
        if classes is None:
            classes = [
                "BUTTOCKS_EXPOSED",
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_GENITALIA_EXPOSED",
                #"MALE_BREAST_EXPOSED",
                "ANUS_EXPOSED",
                "MALE_GENITALIA_EXPOSED"
            ]
        detections = [detection for detection in detections if detection["class"] in classes and detection["score"] > 0.7]
        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # Ensure the bounding box is within the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            if img.shape[2] == 3:
                img[y: y + h, x: x + w] = (0, 0, 0)
            else:
                # Handle grayscale images or other formats
                print("Image has an unexpected number of channels.")
        return img

    def status(self, img, classes=None):
        detections = self.detect(img)
        if classes is None:
            classes = [
                "BUTTOCKS_EXPOSED",
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_GENITALIA_EXPOSED",
                #"MALE_BREAST_EXPOSED",
                "ANUS_EXPOSED",
                "MALE_GENITALIA_EXPOSED"
            ]
        detections = [detection for detection in detections if detection["class"] in classes and detection["score"] > 0.7]
        if len(detections) > 0:
            return False
        return True
