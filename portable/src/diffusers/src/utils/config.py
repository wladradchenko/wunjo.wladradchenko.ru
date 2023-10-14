import os
import json
from typing import Optional, Sequence, Tuple


from .diffusers_videoio import get_frame_count


class RenderConfig:
    input_path = None
    work_dir = None
    key_subdir = None
    output_path = None
    interval = 10
    control_strength = 1
    loose_cfattn = False
    freeu_args = (1, 1, 1, 1)
    use_limit_device_resolution = True
    control_type = "hed"
    canny_low = None
    canny_high = None

    style_update_freq = 10
    mask_strength = 0.5
    color_preserve = True
    mask_period = (0.5, 0.8)
    inner_strength = 0.9
    cross_period = (0, 1)
    ada_period = (1.0, 1.0)
    warp_period = (0, 0.1)
    smooth_boundary = True

    def __init__(self):
        ...

    def update_default_parameters(self,
                                  prompt: str,
                                  a_prompt: str = '',
                                  n_prompt: str = '',
                                  x0_strength: float = -1,
                                  crop: Sequence[int] = (0, 0, 0, 0),
                                  sd_model: Optional[str] = None,
                                  ddim_steps=20,
                                  scale=7.5,
                                  seed: int = -1,
                                  image_resolution: int = 512,
                                  style_update_freq: int = 10,
                                  cross_period: Tuple[float, float] = (0, 1),
                                  warp_period: Tuple[float, float] = (0, 0.1),
                                  mask_period: Tuple[float, float] = (0.5, 0.8),
                                  ada_period: Tuple[float, float] = (1.0, 1.0),
                                  mask_strength: float = 0.5,
                                  inner_strength: float = 0.9,
                                  smooth_boundary: bool = True,
                                  color_preserve: bool = True,
                                  ):
        self.input_path = self.input_path
        self.output_path = self.output_path
        self.prompt = prompt
        self.work_dir = self.work_dir
        self.key_dir = self.key_subdir
        self.first_dir = os.path.join(self.work_dir, 'first')
        os.makedirs(self.first_dir, exist_ok=True)

        # Split video into frames
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(f'Cannot find video file {self.input_path}')
        self.input_dir = os.path.join(self.work_dir, 'video')
        os.makedirs(self.input_dir, exist_ok=True)

        self.frame_count = get_frame_count(self.input_path)
        self.interval = self.interval
        self.crop = crop
        self.sd_model = sd_model
        self.a_prompt = a_prompt
        self.n_prompt = n_prompt
        self.ddim_steps = ddim_steps
        self.scale = scale
        self.control_type = self.control_type
        if self.control_type == 'canny':
            self.canny_low = self.canny_low
            self.canny_high = self.canny_high
        else:
            self.canny_low = None
            self.canny_high = None
        self.control_strength = self.control_strength
        self.seed = seed
        self.image_resolution = image_resolution
        self.use_limit_device_resolution = self.use_limit_device_resolution
        self.x0_strength = x0_strength
        self.style_update_freq = style_update_freq
        self.cross_period = cross_period
        self.mask_period = mask_period
        self.warp_period = warp_period
        self.ada_period = ada_period
        self.mask_strength = mask_strength
        self.inner_strength = inner_strength
        self.smooth_boundary = smooth_boundary
        self.color_preserve = color_preserve
        self.loose_cfattn = self.loose_cfattn
        self.freeu_args = self.freeu_args

    @property
    def use_warp(self):
        return self.warp_period[0] <= self.warp_period[1]

    @property
    def use_mask(self):
        return self.mask_period[0] <= self.mask_period[1]

    @property
    def use_ada(self):
        return self.ada_period[0] <= self.ada_period[1]
