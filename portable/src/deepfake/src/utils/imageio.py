import os
import cv2
import uuid
import numpy as np


def save_image_cv2(save_file, target_frame):
    cv2.imwrite(save_file, target_frame)
    return save_file


def read_image_pil(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes


def read_image_cv2(image_path):
    return cv2.imread(image_path)


def generate_random_color():
    """Generate a random RGB color."""
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))


def save_colored_mask_cv2(save_path, mask, save_name=None):
    save_name = str(uuid.uuid4()) + '.png' if save_name is None else save_name
    # Reduce the dimensions if necessary
    if mask.ndim == 4:
        mask = mask[0, 0]
    # Convert boolean values to 0 or 255
    mask_to_save = (mask * 255).astype(np.uint8)
    # Convert grayscale to BGR
    colored_mask = cv2.cvtColor(mask_to_save, cv2.COLOR_GRAY2BGR)
    # Replace white with random color
    random_color = generate_random_color()
    colored_mask[mask_to_save == 255] = random_color
    # Create alpha channel: 0 for black, 255 for colored regions
    alpha_channel = np.ones(mask_to_save.shape, dtype=mask_to_save.dtype) * 255
    alpha_channel[mask_to_save == 0] = 0
    # Merge RGB and alpha into a 4-channel image (BGRA)
    bgra_mask = cv2.merge((colored_mask[:, :, 0], colored_mask[:, :, 1], colored_mask[:, :, 2], alpha_channel))
    # Save the image
    cv2.imwrite(os.path.join(save_path, save_name), bgra_mask)
    return save_name


def save_mask_image_cv2(save_path, mask):
    save_name = str(uuid.uuid4())+'.png'
    # Reduce the dimensions
    if mask.ndim == 4:
        mask = mask[0, 0]

    # Convert boolean values to 0 or 255
    mask_image = (mask * 255).astype(np.uint8)
    # Save using OpenCV
    cv2.imwrite(os.path.join(save_path, save_name), mask_image)
    return save_name