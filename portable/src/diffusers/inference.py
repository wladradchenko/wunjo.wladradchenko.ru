import os
import sys
from time import strftime
from argparse import Namespace

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
diffusers_root_path = os.path.join(root_path, "diffusers")
sys.path.insert(0, diffusers_root_path)

from src.utils.video2render import render
from src.utils.config import RenderConfig

sys.path.pop(0)

from cog import Input
from backend.folders import TMP_FOLDER, DIFFUSERS_MODEL_FOLDER


class Video2Video:
    @staticmethod
    def main_video_render(source: str, output_folder: str, masks: dict = None, interval: int = 10, source_start: float = 0,
                          source_end: float = 0, source_type: str = "video", control_strength: float = 0.7,
                          use_limit_device_resolution: bool = True, predictor = None, session = None,
                          control_type: str = Input(description="Choose a controlnet", choices=["canny", "hed"], default="canny",),
                          translation: str = Input(description=" Advanced options for the key frame translation", choices=["loose_cfattn", "freeu", None], default="loose_cfattn",),):
        args = Video2Video.load_video2video_default()
        # create config
        cfg = RenderConfig()
        # folders and files
        cfg.input_path = source  # input video
        cfg.work_dir = os.path.join(output_folder, strftime("%Y_%m_%d_%H.%M.%S"))  # output folder
        os.makedirs(cfg.work_dir, exist_ok=True)
        cfg.output_path = os.path.join(cfg.work_dir, "blend.mp4")  # output video
        cfg.key_subdir = os.path.join(cfg.work_dir, "keys")  # frame folder
        # values
        cfg.interval = interval
        cfg.control_strength = control_strength
        if translation == "loose_cfattn":
            cfg.loose_cfattn = True
            cfg.freeu_args = (1, 1, 1, 1)
        elif translation == "freeu":
            cfg.loose_cfattn = False
        else:
            cfg.loose_cfattn = False
            cfg.freeu_args = (1, 1, 1, 1)
        cfg.use_limit_device_resolution = use_limit_device_resolution
        cfg.control_type = control_type  # TODO maybe need to use openpose too?
        cfg.canny_low = None if cfg.control_type != "canny" else cfg.canny_low
        cfg.canny_high = None if cfg.control_type != "canny" else cfg.canny_high
        # image_resolution set from frame TODO
        # model path TODO
        sd_model_path = os.path.join(DIFFUSERS_MODEL_FOLDER, "realisticVisionV20_v20.safetensors")
        # TODO other params get from mask will be
        print(masks)
        # diffusion frames
        render(cfg, args.one, args.key_video_path)

    @staticmethod
    def load_video2video_default():
        return Namespace(
            cfg=None,  # str Path to config file (TODO change on load from frontend)
            input=None,  # str The input path to video.
            output=None,  # str
            prompt=None,  # str
            key_video_path=None,  # str
            one=None,  # bool Run the first frame with ControlNet only
            nr=None,  # bool Do not run rerender and do postprocessing only
            nb=None,  # bool Do not run postprocessing and run rerender only
            ne=None,  # bool Do not run ebsynth (use previous ebsynth temporary output)
            nps=None,  # bool Do not run poisson gradient blending
            n_proc=1,  # int The max process count (TODO maybe control param)
            tmp=None,  # bool Keep ebsynth temporary output,
            a_prompt="RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",  # improve prompt
            n_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",  # default negative prompt
            crop=[0, 0, 0, 0],  # crop video turn off
            canny_low=50,
            canny_high=100,
            warp_period=[0, 0.1],
            ada_period=[0.8, 1],
            freeu_args=[1.1, 1.2, 1.0, 0.2],
        )