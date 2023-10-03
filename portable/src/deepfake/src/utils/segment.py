import os
import sys
import cv2
import numpy as np
import onnxruntime
from PIL import Image

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
deepfake_root_path = os.path.join(root_path, "deepfake")
sys.path.insert(0, deepfake_root_path)
from src.segment_anything import sam_model_registry, SamPredictor
sys.path.pop(0)


class SegmentAnything:
    def __init__(self):
        self.predictor = None
        self.session = None
        # draw frame
        self.draw_obj_frames = {}

    def load_models(self, predictor, session):
        self.predictor = predictor
        self.session = session

    @staticmethod
    def init_vit(vit_path, vit_type, device):
        sam = sam_model_registry[vit_type](checkpoint=vit_path)
        sam.to(device=device)
        return SamPredictor(sam)

    @staticmethod
    def init_onnx(onnx_path, device):
        access_providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in access_providers:
            provider = ["CUDAExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]
        return onnxruntime.InferenceSession(onnx_path, providers=provider)

    @staticmethod
    def get_embedding(predictor, img: np.ndarray):
        predictor.set_image(img)
        return predictor.get_image_embedding().cpu().numpy()

    @staticmethod
    def read_image(img_path: str):
        image = cv2.imread(img_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def draw_mask(predictor, session, point_list, frame):
        originalHeight, originalWidth, _ = frame.shape

        input_point = []
        input_label = []

        for point in point_list:
            canvasWidth = point["canvasWidth"]
            canvasHeight = point["canvasHeight"]

            # Calculate the scale factor
            scaleFactorX = originalWidth / canvasWidth
            scaleFactorY = originalHeight / canvasHeight

            # Calculate the new position and size of the points on the original image
            newX = int(point['x'] * scaleFactorX)
            newY = int(point['y'] * scaleFactorY)

            # Append the newX and newY to input_point
            input_point.append([newX, newY])

            # Transfer label to color and append to input_labels
            if point.get("color") == "lightblue":
                input_label.append(1)
            elif point.get("color") == "red":
                input_label.append(0)
            else:
                print("Color from point_list is not default")

        input_point = np.array(input_point)
        input_label = np.array(input_label)

        predictor.set_image(frame)
        frame_embedding = predictor.get_image_embedding().cpu().numpy()

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = predictor.transform.apply_coords(onnx_coord, frame.shape[:2]).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)  # if low_res_logits is None else low_res_logits
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": frame_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(frame.shape[:2], dtype=np.float32)
        }

        masks, _, low_res_logits = session.run(None, ort_inputs)
        masks = masks > predictor.model.mask_threshold

        return masks
