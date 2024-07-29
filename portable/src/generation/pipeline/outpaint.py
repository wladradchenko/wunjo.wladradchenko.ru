import cv2
import numpy as np


class Outpaint:
    @staticmethod
    def merge_images(original, new_image, offset, direction):
        if direction in ["left", "right"]:
            merged_image = np.zeros((original.shape[0], original.shape[1] + offset, 3), dtype=np.uint8)
        elif direction in ["top", "bottom"]:
            merged_image = np.zeros((original.shape[0] + offset, original.shape[1], 3), dtype=np.uint8)

        if direction == "left":
            merged_image[:, offset:] = original
            merged_image[:, : new_image.shape[1]] = new_image
        elif direction == "right":
            merged_image[:, : original.shape[1]] = original
            merged_image[:, original.shape[1] + offset - new_image.shape[1] : original.shape[1] + offset] = new_image
        elif direction == "top":
            merged_image[offset:, :] = original
            merged_image[: new_image.shape[0], :] = new_image
        elif direction == "bottom":
            merged_image[: original.shape[0], :] = original
            merged_image[original.shape[0] + offset - new_image.shape[0] : original.shape[0] + offset, :] = new_image

        return merged_image

    @staticmethod
    def process_image(image, fill_color=(0, 0, 0), mask_offset=50, blur_radius=500, expand_pixels=256, direction="left",
                      inpaint_mask_color=50, max_size=1024):

        height, width = image.shape[:2]

        new_height = height + (expand_pixels if direction in ["top", "bottom"] else 0)
        new_width = width + (expand_pixels if direction in ["left", "right"] else 0)

        if new_height > max_size:
            # If so, crop the image from the opposite side
            if direction == "top":
                image = image[:max_size, :]
            elif direction == "bottom":
                image = image[new_height - max_size :, :]
            new_height = max_size

        if new_width > max_size:
            # If so, crop the image from the opposite side
            if direction == "left":
                image = image[:, :max_size]
            elif direction == "right":
                image = image[:, new_width - max_size :]
            new_width = max_size

        height, width = image.shape[:2]

        new_image = np.full((new_height, new_width, 3), fill_color, dtype=np.uint8)
        mask = np.full_like(new_image, 255, dtype=np.uint8)
        inpaint_mask = np.full_like(new_image, 0, dtype=np.uint8)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        inpaint_mask = cv2.cvtColor(inpaint_mask, cv2.COLOR_BGR2GRAY)

        if direction == "left":
            new_image[:, expand_pixels:] = image[:, : max_size - expand_pixels]
            mask[:, : expand_pixels + mask_offset] = inpaint_mask_color
            inpaint_mask[:, :expand_pixels] = 255
        elif direction == "right":
            new_image[:, :width] = image
            mask[:, width - mask_offset :] = inpaint_mask_color
            inpaint_mask[:, width:] = 255
        elif direction == "top":
            new_image[expand_pixels:, :] = image[: max_size - expand_pixels, :]
            mask[: expand_pixels + mask_offset, :] = inpaint_mask_color
            inpaint_mask[:expand_pixels, :] = 255
        elif direction == "bottom":
            new_image[:height, :] = image
            mask[height - mask_offset :, :] = inpaint_mask_color
            inpaint_mask[height:, :] = 255

        # mask blur
        if blur_radius % 2 == 0:
            blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        # telea inpaint
        _, mask_np = cv2.threshold(inpaint_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        inpaint = cv2.inpaint(new_image, mask_np, 3, cv2.INPAINT_TELEA)

        return inpaint, mask

    @staticmethod
    def slice_image(image):
        height, width, _ = image.shape
        slice_size = min(width // 2, height // 3)

        slices = []

        for h in range(3):
            for w in range(2):
                left = w * slice_size
                upper = h * slice_size
                right = left + slice_size
                lower = upper + slice_size

                if w == 1 and right > width:
                    left -= right - width
                    right = width
                if h == 2 and lower > height:
                    upper -= lower - height
                    lower = height

                slice = image[upper:lower, left:right]
                slices.append(slice)

        return slices

    @staticmethod
    def image_resize_max(image, max_height=576, max_width=1024):
        height, width = image.shape[:2]

        aspect_ratio = width / height

        # Resize based on height limit
        if height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        # Resize based on width limit
        elif width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        # No resize needed
        else:
            new_width = width
            new_height = height

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return resized_image

    def decode_image(self, img_path, max_height=576, max_width=1024, inpaint_mask_color=50, padding=20):
        original = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if original.shape[2] == 4:
            # Image has an alpha channel; remove it or process it
            original = original[:, :, :3]  # Remove alpha channel

        if original.shape[0] == max_height and original.shape[1] == max_width:
            return original, np.zeros((max_height, max_width), dtype=np.uint8)

        image = self.image_resize_max(original, max_height, max_width)
        height, width, channels = image.shape

        expand_pixels_half_x = int((max_width - width) / 2)
        expand_pixels_half_y = int((max_height - height) / 2)

        if expand_pixels_half_x > 0:
            expand_pixels_half = expand_pixels_half_x
            directions = ["left", "right"]
        else:
            expand_pixels_half = expand_pixels_half_y
            directions = ["top", "bottom"]

        generated_images = []
        masks = []

        for direction in directions:
            image_part, mask = self.process_image(
                image,
                expand_pixels=expand_pixels_half,
                direction=direction,
                inpaint_mask_color=inpaint_mask_color,
                max_size = max_height if max_height > max_width else max_width
            )
            generated_images.append(image_part)
            masks.append(mask)

        # Merge generated images
        output_image = generated_images[0]
        for i in range(1, len(generated_images)):
            output_image = self.merge_images(output_image, generated_images[i], expand_pixels_half, directions[i])

        # Create a binary mask with padding
        mask_combined = np.zeros((max_height, max_width), dtype=np.uint8) + 255

        center_x = max_width // 2
        center_y = max_height // 2

        if expand_pixels_half_x > 0:
            pad_x_start = center_x - width // 2 + padding
            pad_x_end = pad_x_start + width - padding * 2
            pad_y_start = center_y - height // 2
            pad_y_end = pad_y_start + height
        else:
            pad_x_start = center_x - width // 2
            pad_x_end = pad_x_start + width
            pad_y_start = center_y - height // 2 + padding
            pad_y_end = pad_y_start + height - padding * 2

        mask_combined[pad_y_start:pad_y_end, pad_x_start:pad_x_end] = 0

        return output_image, mask_combined