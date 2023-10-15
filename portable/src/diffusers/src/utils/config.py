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

    @property
    def use_warp(self):
        return self.warp_period[0] <= self.warp_period[1]

    @property
    def use_mask(self):
        return self.mask_period[0] <= self.mask_period[1]

    @property
    def use_ada(self):
        return self.ada_period[0] <= self.ada_period[1]
