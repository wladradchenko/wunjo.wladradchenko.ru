import os
import math
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm


class YolaSegment:
    """Adaptation Yola segment model to ONNX with dynamic and static size"""
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32, device='cpu'):
        """
        Init
        :param path: path to model
        :param conf_thres: fond thres
        :param iou_thres: iou thres
        :param num_masks: num_masks
        :param device: device
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.device = device

        # Initialize model
        self.initialize_model(path)

        # Utils
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Create a list of colors for each class where each color is a tuple of 3 integer values
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))

    def __call__(self, image):
        """
        Forward
        :param image: frame
        :return:
        """
        return self.segment_objects(image)

    def initialize_model(self, path):
        """
        Init model with device
        :param path: path to model
        :return:
        """
        access_providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in access_providers:
            provider = ["CUDAExecutionProvider"] if self.device == 'cuda' else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        self.session = onnxruntime.InferenceSession(path, providers=provider)
        # Get model info
        self.get_input_details()
        self.get_output_details()

    @staticmethod
    def xywh2xyxy(x):
        """
        Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        :param x: x, y, w, h
        :return:
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    @staticmethod
    def compute_iou(box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def segment_objects(self, image):
        """
        Get segmentation
        :param image: frame
        :return:
        """
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image, divisor=32):
        """
        Prepare input to tensor with adaptive size for dynamic and static models
        :param image: frame
        :param divisor: divisor using only for model dynamic size
        :return:
        """
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(self.input_width, str) or isinstance(self.input_height, str):
            # Make the size divisible by the divisor
            self.input_width = (self.img_width + divisor - 1) // divisor * divisor
            self.input_height = (self.img_height + divisor - 1) // divisor * divisor

        # Resize input image to the new height and width
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        """
        Inference
        :param input_tensor: input tensor
        :return:
        """
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes, (self.img_height, self.img_width), (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1),  interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, (self.input_height, self.input_width), (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
        """
        Draw detection as box or mask
        :param image: frame
        :param boxes: box for object
        :param scores: score if wanna draw
        :param class_ids: class
        :param mask_alpha: mask alpha
        :param mask_maps: mask maps for segmentation else will draw box
        :return:
        """
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = self.draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

            cv2.putText(mask_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return mask_img

    def draw_masks(self, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
        """
        Draw segmentation mask
        :param image: frame
        :param boxes: boxe
        :param class_ids: class
        :param mask_alpha: mask alpha
        :param mask_maps:
        :return: mask maps for segmentation else will draw box
        """
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill mask image
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    def box_detection_mode(self, image, mask_alpha=0.4):
        """
        Simple draw only all box on frame
        :param image: frame
        :param mask_alpha: mask alpha
        :return:
        """
        return self.draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def segment_detection_mode(self, image, mask_alpha=0.5):
        """
        Simple all segmentation draw on frame
        :param image: frame
        :param mask_alpha: mask alpha
        :return:
        """
        return self.draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def segment_convert_img_mask(self, input_path, output_path, labels=None, color_rgb=None):
        input_format_img = ['png', 'jpg', 'jpeg']

        if os.path.isfile(input_path):
            files = [input_path] if os.path.basename(input_path).split('.')[-1] in input_format_img else []
        else:
            files = os.listdir(input_path)
            # Sort files by their integer values
            try:
                files = sorted(files, key=lambda x: int(x.split('.')[0]))
                files = [os.path.join(input_path, filename) for filename in files if filename.split('.')[-1] in input_format_img]
            except Exception as err:
                print("In segment convert img mask files doesn't have number name")
                files = [os.path.join(input_path, filename) for filename in os.listdir(input_path) if filename.split('.')[-1] in input_format_img]

        bar = tqdm(total=len(files) * len(labels), dynamic_ncols=True)

        for num, img_path in enumerate(files):
            img_name = os.path.basename(img_path)
            save_frame = os.path.join(output_path, img_name.split('.')[0])
            os.makedirs(save_frame, exist_ok=True)

            img = cv2.imread(img_path)
            h, w, _ = img.shape

            # Detect Objects
            boxes, scores, class_ids, masks = self.segment_objects(img)
            mask_img = np.zeros((h, w, 3), dtype=np.float32)
            for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
                if self.class_names[class_id] in labels or labels is None:
                    local_mask_img = np.zeros((h, w, 3), dtype=np.float32)

                    color = self.colors[class_id] if color_rgb is None else color_rgb
                    x1, y1, x2, y2 = box.astype(int)

                    crop_mask = self.mask_maps[i][y1:y2, x1:x2, np.newaxis]
                    crop_mask_img = local_mask_img[y1:y2, x1:x2]  # local mask source
                    crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                    mask_img[y1:y2, x1:x2] = crop_mask_img  # general mask result
                    local_mask_img[y1:y2, x1:x2] = crop_mask_img  # local mask result

                    if labels is not None:
                        cv2.imwrite(os.path.join(save_frame, f"{self.class_names[class_id]}_{i}_{img_name}"), local_mask_img)
            cv2.imwrite(os.path.join(save_frame, f"mask_{img_name}"), mask_img)
            # Reverse the alpha channel
            scaled_mask_img = mask_img / 255.0
            # Reverse the mask
            reversed_mask = 1 - scaled_mask_img
            # Rescale the reversed mask to 0-255 range
            reversed_mask_255 = (reversed_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_frame, f"reverse_mask_{img_name}"), reversed_mask_255)
            bar.update(1)

        bar.close()
        return output_path

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        """
        Rescale boxes to original image dimensions
        :param boxes: boxe
        :param input_shape: current shape
        :param image_shape: original shape
        :return:
        """
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes
