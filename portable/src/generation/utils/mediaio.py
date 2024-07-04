import numpy as np
from PIL import Image


def resize_pil_image64(image: Image.Image, max_size: int):
    """
    Resize the image to ensure its width and height do not exceed max_size while maintaining the aspect ratio.

    Args:
    image (Image.Image): The input image to be resized.
    max_size (int): The maximum allowed size for both width and height.

    Returns:
    tuple: A tuple containing the original width and height of the image.
    """
    original_width, original_height = image.size

    if original_width > max_size or original_height > max_size:
        if original_width > original_height:
            new_width = max_size
            new_height = int((max_size / original_width) * original_height)
        else:
            new_height = max_size
            new_width = int((max_size / original_height) * original_width)

        new_height = int(np.round(new_height / 64.0)) * 64
        new_width = int(np.round(new_width / 64.0)) * 64

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        new_height = int(np.round(original_height / 64.0)) * 64
        new_width = int(np.round(original_width / 64.0)) * 64

        if new_height != original_height or new_width != original_width:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert image to JPEG format
    image = image.convert('RGB')

    return original_width, original_height, image
