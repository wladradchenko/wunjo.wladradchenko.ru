# coding: utf-8

"""
parameters used for crop faces
"""

from dataclasses import dataclass


@dataclass
class CropConfig:
    insightface_root: str = None
    landmark_ckpt_path: str = None
    # TODO понять, почему в LivePortrait косичка двигается, а у меня кроп меньше после paste_back

    device_id: int = 0  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP
    det_thresh: float = 0.1 # detection threshold
    ########## source image or video cropping option ##########
    dsize: int = 512  # crop size
    scale: float = 2.3  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    max_face_num: int = 0  # max face number, 0 mean no limit
    flag_do_rot: bool = True # whether to conduct the rotation when flag_do_crop is True
    animal_face_type: str = "animal_face_9"  # animal_face_68 -> 68 landmark points, animal_face_9 -> 9 landmarks
    ########## driving video auto cropping option ##########
    scale_crop_driving_video: float = 2.2  # 2.0 # scale factor for cropping driving video
    vx_ratio_crop_driving_video: float = 0.0  # adjust y offset
    vy_ratio_crop_driving_video: float = -0.1  # adjust x offset
    direction: str = "large-small"  # direction of cropping
    crop: dict = None
