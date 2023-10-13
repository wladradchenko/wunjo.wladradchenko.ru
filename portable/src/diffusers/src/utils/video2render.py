import cv2
import numpy as np
from PIL import Image
from skimage import exposure


from .blend import BlendType, blendLayers  # TODO check work this or not
from .config import RenderConfig


def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(cv2.cvtColor(np.asarray(original_image),  cv2.COLOR_RGB2LAB), correction, channel_axis=2), cv2.COLOR_LAB2RGB).astype('uint8'))
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)
    return image


def render(cfg: RenderConfig, first_img_only: bool, key_video_path: str):
    pass