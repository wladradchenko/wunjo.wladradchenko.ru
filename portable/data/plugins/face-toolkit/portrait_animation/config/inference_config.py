# coding: utf-8

"""
config dataclass used for inference
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Tuple


def create_soft_square(size=512, border_thickness=15, blur_radius=40):
    # Create a blank black image
    img = np.zeros((size, size), dtype=np.uint8)
    # Define the size of the rectangle inside the image
    top_left = (border_thickness, border_thickness)
    bottom_right = (size - border_thickness, size - border_thickness)
    # Draw the rounded rectangle on the black image
    cv2.rectangle(img, top_left, bottom_right, 255, -1)  # Solid white rectangle
    # Create a blurred version for soft edges
    # mask blur
    if blur_radius % 2 == 0:
        blur_radius += 1
    blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
    # Optionally convert to color (RGB) format if needed
    blurred_color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    return blurred_color


def get_lip_array():
    lip_array = np.array([[[-9.45240e-05, 2.42220e-04, -4.00974e-05],
                           [2.08596e-04, 4.68054e-04, 1.27464e-05],
                           [-1.24602e-04, 3.24972e-04, 4.42500e-06],
                           [1.19652e-04 - 4.56624e-04, 5.40852e-05],
                           [-2.65254e-04 - 7.45980e-05, -1.55280e-05],
                           [-3.34884e-05 - 1.83036e-04, -3.49404e-06],
                           [-4.75428e-07 - 4.52466e-07, 1.59498e-06],
                           [-5.41296e-04, 2.63280e-03, 7.24860e-06],
                           [-1.16940e-05, 5.00502e-06, 9.98880e-06],
                           [8.02860e-07, 5.19132e-06, -4.09344e-04],
                           [1.66878e-04, -1.98150e-05, -8.85480e-05],
                           [-1.51536e-05, 8.25060e-05, 5.94012e-06],
                           [-2.61054e-07, 2.75910e-06, 2.21028e-06],
                           [-6.62820e-05, -6.82860e-05, -3.45318e-06],
                           [-8.32620e-04, -1.79592e-03, -2.59176e-06],
                           [6.97140e-06, 3.65514e-05, -1.89780e-06],
                           [-4.80876e-06, 7.11480e-06, -1.03236e-05],
                           [-6.36840e-04, -1.16610e-03, 5.62314e-05],
                           [2.83290e-04, -1.45656e-05, 1.09746e-05],
                           [-4.16022e-06, -1.50906e-0, -2.54940e-03],
                           [2.22474e-04, 5.62788e-03, 2.72646e-03]]], dtype=object)
    return lip_array


@dataclass
class InferenceConfig:
    # HUMAN MODEL CONFIG, NOT EXPORTED PARAMS
    checkpoint_F: str = None  # path to checkpoint of F
    checkpoint_M: str = None  # path to checkpoint pf M
    checkpoint_G: str = None  # path to checkpoint of G
    checkpoint_W: str = None  # path to checkpoint of W
    checkpoint_S: str = None  # path to checkpoint to S and R_eyes, R_lip

    # EXPORTED PARAMS
    flag_use_half_precision: bool = True
    flag_crop_driving_video: bool = False
    device_id: int = 0
    flag_normalize_lip: bool = True
    flag_source_video_eye_retargeting: bool = False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_do_rot: bool = True
    flag_force_cpu: bool = False
    flag_do_torch_compile: bool = False
    driving_option: str = "pose-friendly"  # "expression-friendly" or "pose-friendly"
    driving_multiplier: float = 1.0
    driving_smooth_observation_variance: float = 3e-7  # smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy
    source_max_dim: int = 1280  # the max dim of height and width of source image or video
    source_division: int = 2  # make sure the height and width of source image or video can be divided by this number
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all"  # the region where the animation was performed, "exp" means the expression, "pose" means the head pose

    # NOT EXPORTED PARAMS
    lip_normalize_threshold: float = 0.03  # threshold for flag_normalize_lip
    source_video_eye_retargeting_threshold: float = 0.18  # threshold for eyes retargeting if the input is a source video
    anchor_frame: int = 0  # TO IMPLEMENT

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    crf: int = 15  # crf for output video
    output_fps: int = 25 # default output fps

    mask_crop: np.ndarray = field(default_factory=lambda: create_soft_square())
    lip_array: np.ndarray = field(default_factory=get_lip_array)
    size_gif: int = 256 # default gif size, TO IMPLEMENT

    # Added for webcam
    ref_max_shape: int = 1280
    ref_shape_n: int = 2

    flag_lip_zero: bool = True  # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    lip_zero_threshold: float = 0.03
