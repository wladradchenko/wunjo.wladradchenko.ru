import os
import sys
import cv2
import random
import numpy as np
import onnxruntime
from PIL import Image

from deepfake.src.segment_anything import sam_model_registry, SamPredictor


class SegmentAnything:
    def __init__(self, segment_percentage: float = 0.25):
        self.predictor = None
        self.session = None
        # draw frame
        self.draw_obj = {}
        self.lower_limit_area = 1 - segment_percentage  # 25% lower
        self.upper_limit_area = 1 + segment_percentage  # 25% upper
        self.generate_num_positive_points = 4

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
    def draw_mask(predictor, session, point_list, frame, box=None):
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

        if box is not None:
            onnx_box_coords = box.reshape(2, 2)
            onnx_box_labels = np.array([2, 3])

            onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)
        else:
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = predictor.transform.apply_coords(onnx_coord, frame.shape[:2]).astype(np.float32)

        predictor.set_image(frame)
        frame_embedding = predictor.get_image_embedding().cpu().numpy()

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": frame_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(frame.shape[:2], dtype=np.float32)
        }

        masks, _, _ = session.run(None, ort_inputs)
        masks = masks > predictor.model.mask_threshold
        # TODO if it will be very slow, I can optimize by keeping low_res_logits in self for each frame to nor repeat load
        return masks

    def draw_mask_frames(self, frame):
        """
        Predict mask and move draw_obj for objid
        :param frame: frame
        :return: list mask in format false true
        """
        originalHeight, originalWidth, _ = frame.shape
        point_list = self.draw_obj["point_list"]
        cX_prev = self.draw_obj["cX"]
        cY_prev = self.draw_obj["cY"]
        prev_area = self.draw_obj["area"]
        prev_box = self.draw_obj["box"]
        mask = self.draw_mask(predictor=self.predictor, session=self.session, point_list=point_list, frame=frame, box=prev_box)
        centroid = self.compute_centroid(mask)
        if centroid is None:
            cX = cX_prev
            cY = cY_prev
        else:
            cX, cY = centroid
        area = self.calculate_mask_area(mask)
        if self.lower_limit_area >= area / prev_area or area / prev_area >= self.upper_limit_area:
            return None
        for point in point_list:
            canvasWidth = point["canvasWidth"]
            canvasHeight = point["canvasHeight"]
            # Calculate the scale factor
            scaleFactorX = originalWidth / canvasWidth
            scaleFactorY = originalHeight / canvasHeight
            # Update coordinate for point as obj can move
            point["x"] = int(point["x"] + (cX - cX_prev) / scaleFactorX)
            point["y"] = int(point["y"] + (cY - cY_prev) / scaleFactorY)
        box = self.get_max_bounding_box(mask)
        self.draw_obj = {"point_list": point_list, "cX": cX, "cY": cY, "area": area, "box": box}
        return mask

    @staticmethod
    def compute_centroid(mask):
        """
        Compute the centroid of the largest contour in the mask.
        :param mask: A 2D numpy array where non-zero values represent the mask.
        returns:  tuple (cX, cY) representing the centroid of the largest contour.
        """
        if mask.ndim == 4:
            mask = mask[0, 0]
        # Convert boolean mask to 0 or 255
        mask = (mask * 255).astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If no contours are found, return None
        if not contours:
            return None
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        # Compute the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return None

    @staticmethod
    def calculate_mask_area(mask):
        """
        Calculate the area of the mask.
        param mask: A numpy array where non-zero values represent the mask. Can be 2D or 4D, where the relevant slice is [0, 0].
        return: The area of the mask (number of non-zero pixels).
        """
        # Handle 4D mask by extracting the relevant 2D slice
        if mask.ndim == 4:
            mask = mask[0, 0]
        # Count the number of non-zero pixels
        area = np.count_nonzero(mask)
        return area


    @staticmethod
    def get_max_bounding_box(mask):
        """
        Get the bounding box with the maximum area from the mask.
        :param mask: A 2D numpy array where `True` values represent the mask.
        returns: Bounding box in the format [x1, y1, x2, y2].
        """
        if mask.ndim == 4:
            mask = mask[0, 0]
        # Convert boolean mask to 0 or 255
        mask_255 = (mask * 255).astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_bbox = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > max_area:
                max_area = area
                max_bbox = [x, y, x + w, y + h]
        return np.array(max_bbox)

    def set_obj(self, point_list, frame):
        mask = self.draw_mask(predictor=self.predictor, session=self.session, point_list=point_list, frame=frame)
        centroid = self.compute_centroid(mask)
        if centroid is None:
            cX = 0
            cY = 0
        else:
            cX, cY = centroid
        area = self.calculate_mask_area(mask)
        box = self.get_max_bounding_box(mask)
        self.draw_obj = {"point_list": point_list, "cX": cX, "cY": cY, "area": area, "box": box}
        return mask

    @staticmethod
    def convert_colored_mask_cv2(mask, mask_color):
        # Reduce the dimensions if necessary
        if mask.ndim == 4:
            mask = mask[0, 0]
        # Convert boolean values to 0 or 255
        mask_to_save = (mask * 255).astype(np.uint8)
        # Convert grayscale to BGR
        colored_mask = cv2.cvtColor(mask_to_save, cv2.COLOR_GRAY2BGR)
        # Replace white with random color
        random_color = (0, 0, 255)
        colored_mask[mask_to_save == 255] = random_color
        # Create alpha channel: 0 for black, 255 for colored regions
        alpha_channel = np.ones(mask_to_save.shape, dtype=mask_to_save.dtype) * 255
        alpha_channel[mask_to_save == 0] = 0
        # Merge RGB and alpha into a 4-channel image (BGRA)
        bgra_mask = cv2.merge((colored_mask[:, :, 0], colored_mask[:, :, 1], colored_mask[:, :, 2], alpha_channel))
        # Convert to PIL
        # Convert the BGRA mask to RGBA
        rgba_mask = bgra_mask[..., [2, 1, 0, 3]]
        # Convert the RGBA mask to a Pillow image
        pil_image = Image.fromarray(rgba_mask, 'RGBA')
        # Or grayscale
        grayscale_image = pil_image.convert("L")
        return grayscale_image

    @staticmethod
    def convert_colored_mask_thickness_cv2(mask):
        # Reduce the dimensions if necessary
        if mask.ndim == 4:
            mask = mask[0, 0]
        # Convert boolean values to 0 or 255
        mask_to_save = (mask * 255).astype(np.uint8)

        # Dilation to increase line thickness
        kernel_size = 10  # Thickness line weight
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_to_save = cv2.dilate(mask_to_save, kernel, iterations=1)

        # Convert grayscale to BGR
        colored_mask = cv2.cvtColor(mask_to_save, cv2.COLOR_GRAY2BGR)
        # Replace white with random color
        random_color = (0, 0, 255)
        colored_mask[mask_to_save == 255] = random_color
        # Create alpha channel: 0 for black, 255 for colored regions
        alpha_channel = np.ones(mask_to_save.shape, dtype=mask_to_save.dtype) * 255
        alpha_channel[mask_to_save == 0] = 0
        # Merge RGB and alpha into a 4-channel image (BGRA)
        bgra_mask = cv2.merge((colored_mask[:, :, 0], colored_mask[:, :, 1], colored_mask[:, :, 2], alpha_channel))
        # Convert to PIL
        # Convert the BGRA mask to RGBA
        rgba_mask = bgra_mask[..., [2, 1, 0, 3]]
        # Convert the RGBA mask to a Pillow image
        pil_image = Image.fromarray(rgba_mask, 'RGBA')
        # Or grayscale
        grayscale_image = pil_image.convert("L")
        return grayscale_image


    @staticmethod
    def apply_mask_on_frame(mask, frame, color, width=None, height=None):
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reduce the dimensions if necessary
        if mask.ndim == 4:
            mask = mask[0, 0]

        if width is not None and height is not None:
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

        # Convert mask's False values to 0 and other values to 255
        mask_to_save = (mask * 255).astype(np.uint8)

        # Create an empty colored image with the same dimensions as the mask
        colored_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        # If the color is transparent (we'll assume an RGBA tuple where A=0 means fully transparent)
        if color[3] == 0:
            colored_img[..., 3] = 0  # Set alpha channel to 0 for full transparency
        else:
            # Set the RGB channels to the specified color
            colored_img[mask_to_save == 0, :3] = color[:3]
            colored_img[mask_to_save == 0, 3] = 255  # Set alpha channel to 255 for non-transparent regions

        # Convert frame to RGBA format
        frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
        # Where the mask is True, replace with the frame values
        colored_img[mask_to_save == 255] = frame_rgba[mask_to_save == 255]
        # Convert to PIL Image
        pil_image = Image.fromarray(colored_img, 'RGBA')

        return pil_image

    @staticmethod
    def save_black_mask(save_name, mask, mask_save_path, width=None, height=None, kernel_size=10):
        # Reduce the dimensions if necessary
        if mask.ndim == 4:
            mask = mask[0, 0]

        if width is not None and height is not None:
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert mask's False values to 0 and other values to 255
        mask_to_save = (mask * 255).astype(np.uint8)
        # Dilation to increase line thickness
        # Thickness line weight
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_to_save = cv2.dilate(mask_to_save, kernel, iterations=1)
        # Create an empty black and white image with the same dimensions as the mask
        bw_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # Where the mask is True, set to white
        bw_img[mask_to_save == 255] = (255, 255, 255)
        # Convert to PIL Image
        pil_image = Image.fromarray(bw_img, 'RGB')
        pil_image.save(os.path.join(mask_save_path, save_name))


    @staticmethod
    def save_white_background_mask(save_path, mask, background_mask_frame, width=None, height=None, kernel_size=10):
        # Ensure both the masks have the same shape
        if background_mask_frame.shape[:2] != mask.shape[:2]:
            background_mask_frame = cv2.resize(background_mask_frame, (width, height))
        # Reduce the dimensions if necessary
        if mask.ndim == 4:
            mask = mask[0, 0]

        if width is not None and height is not None:
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert mask's False values to 0 and other values to 255
        mask_to_save = (mask * 255).astype(np.uint8)
        # Dilation to increase line thickness
        # Thickness line weight
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_to_save = cv2.dilate(mask_to_save, kernel, iterations=1)
        # Invert the mask
        inverted_mask = cv2.bitwise_not(mask_to_save)
        # Where the inverted mask
        background_mask_frame[inverted_mask == 0, :] = (0, 0, 0)
        # Convert to PIL Image
        pil_image = Image.fromarray(background_mask_frame, 'RGB')
        # Save the combined mask
        pil_image.save(os.path.join(save_path))


    @staticmethod
    def hex_to_rgba(color):
        # If the color is transparent, return (0, 0, 0, 0) for RGBA
        if color == "transparent":
            return (0, 0, 0, 0)

        # Convert hex to RGB
        rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]

        # Return as RGBA with full opacity
        return tuple(rgb) + (255,)

