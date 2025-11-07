# coding: utf-8

import os.path as osp
import torch
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from PIL import Image
from typing import List, Tuple, Union
from dataclasses import dataclass, field

from visual_processing.portrait_animation.config.crop_config import CropConfig
from visual_processing.face_detection.recognition import FaceRecognition
from .crop import (
    average_bbox_lst,
    crop_image,
    crop_image_by_bbox,
    parse_bbox_from_landmark,
)
from .io import contiguous, load_image_rgb, update_crop
from .human_landmark_runner import LandmarkRunner as HumanLandmark

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    M_c2o_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # M_c2o list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        self.crop_cfg: CropConfig = kwargs.get("crop_cfg", None)

        device_id = self.crop_cfg.device_id
        flag_force_cpu = self.crop_cfg.flag_force_cpu

        if flag_force_cpu:
            device = "cpu"
            face_analysis_wrapper_provider = ["CPUExecutionProvider"]
        else:
            try:
                if torch.backends.mps.is_available():
                    # Shape inference currently fails with CoreMLExecutionProvider
                    # for the retinaface model
                    device = "mps"
                    face_analysis_wrapper_provider = ["CPUExecutionProvider"]
                else:
                    device = "cuda"
                    face_analysis_wrapper_provider = ["CUDAExecutionProvider"]
            except Exception as err:
                device = "cuda"
                face_analysis_wrapper_provider = ["CUDAExecutionProvider"]

        self.face_analysis_wrapper = FaceRecognition(
                    name="buffalo_l",
                    root=self.crop_cfg.insightface_root,
                    providers=face_analysis_wrapper_provider,
                )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512), det_thresh=self.crop_cfg.det_thresh)
        self.face_analysis_wrapper.warmup()

        self.human_landmark_runner = HumanLandmark(
            ckpt_path=self.crop_cfg.landmark_ckpt_path,
            onnx_provider=device,
            device_id=device_id,
        )
        self.human_landmark_runner.warmup()

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.crop_cfg, k):
                setattr(self.crop_cfg, k, v)

    def crop_single_image(self, obj, **kwargs):
        direction = kwargs.get('direction', 'large-small')

        # crop and align a single image
        if isinstance(obj, str):
            img_rgb = load_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            img_rgb = obj
        else:
            raise Exception("No right image format!")

        src_face = self.face_analysis_wrapper.get(
            img_rgb,
            flag_do_landmark_2d_106=True,
            direction=direction
        )

        if len(src_face) == 0:
            print('No face detected in the source image.')
            raise Exception("No face detected in the source image!")
        elif len(src_face) > 1:
            print(f'More than one face detected in the image, only pick one face by rule {direction}.')

        src_face = src_face[0]
        pts = src_face.landmark_2d_106

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=kwargs.get('dsize', 512),
            scale=kwargs.get('scale', 2.3),
            vy_ratio=kwargs.get('vy_ratio', -0.15),
        )
        # update a 256x256 version for network input or else
        ret_dct['img_crop_256x256'] = cv2.resize(ret_dct['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / kwargs.get('dsize', 512)

        recon_ret = self.human_landmark_runner.run(img_rgb, pts)  # TODO ??? звяем нужен файл landmark runner?
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct

    def calc_lmk_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig, idx: int = 0):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        src_face = self.face_analysis_wrapper.get(
            img_bgr,
            flag_do_landmark_2d_106=True,
            direction=crop_cfg.direction,
            max_face_num=crop_cfg.max_face_num,
            crop=crop_cfg.crop,
            id=idx
        )

        if len(src_face) == 0:
            print("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            print(f"More than one face detected in the image, only pick one face by rule {crop_cfg.direction}.")

        src_face = src_face[0]

        lmk = src_face.landmark_2d_106
        lmk = self.human_landmark_runner.run(img_rgb, lmk)

        return lmk

    def clear_face(self):
        self.face_analysis_wrapper.clear()

    def update_center_face(self, crop):
        self.face_analysis_wrapper.update_center(crop)

    def is_single_centered_face(self, img_rgb_: np.ndarray) -> bool:
        # Copy the source image and convert to BGR for face detection
        img_rgb = img_rgb_.copy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Detect face
        src_face = self.face_analysis_wrapper.get(
            img_bgr,
            flag_do_landmark_2d_106=True,
            direction='large-small',
            max_face_num=0,
        )

        # Check if exactly one face is detected
        if len(src_face) != 1:
            return False

        # Calculate face height as a proportion of image height
        face_bbox = src_face[0].bbox
        face_height = (face_bbox[3] - face_bbox[1]) / img_rgb.shape[0]

        # Check if the face height is greater than 40%
        if face_height < 0.35:
            return False

        # Get face center coordinates
        face_center_x = (face_bbox[0] + face_bbox[2]) // 2
        face_center_y = (face_bbox[1] + face_bbox[3]) // 2

        # Define the boundaries of the central section (3x3 grid)
        grid_left = img_rgb.shape[1] // 3
        grid_right = 2 * (img_rgb.shape[1] // 3)
        grid_top = img_rgb.shape[0] // 3
        grid_bottom = 2 * (img_rgb.shape[0] // 3)

        # Check if the face center is within the central section of the image
        if grid_left < face_center_x < grid_right and grid_top < face_center_y < grid_bottom:
            return True

        return False

    def crop_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig, idx: int = 0):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        src_face = self.face_analysis_wrapper.get(
            img_bgr,
            flag_do_landmark_2d_106=True,
            direction=crop_cfg.direction,
            max_face_num=crop_cfg.max_face_num,
            crop=crop_cfg.crop,
            id=idx
        )

        if len(src_face) == 0:
            print("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            print(f"More than one face detected in the image, only pick one face by rule {crop_cfg.direction}.")

        # NOTE: temporarily only pick the first face, to support multiple face in the future
        src_face = src_face[0]
        lmk = src_face.landmark_2d_106  # this is the 106 landmarks from insightface

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=crop_cfg.dsize,
            scale=crop_cfg.scale,
            vx_ratio=crop_cfg.vx_ratio,
            vy_ratio=crop_cfg.vy_ratio,
            flag_do_rot=crop_cfg.flag_do_rot,
        )

        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

        lmk = self.human_landmark_runner.run(img_rgb, lmk)
        ret_dct["lmk_crop"] = lmk
        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize

        return ret_dct

    def crop_driving_video(self, driving_rgb_lst, crop_cfg: CropConfig, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        idx_detect = kwargs.get("idx", 0)
        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                    crop=crop_cfg.crop,
                    id=idx_detect
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    print(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {crop_cfg.direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=crop_cfg.scale_crop_driving_video,
                vx_ratio_crop_driving_video=crop_cfg.vx_ratio_crop_driving_video,
                vy_ratio=crop_cfg.vy_ratio_crop_driving_video,
            )["bbox"]
            bbox = [
                ret_bbox[0, 0],
                ret_bbox[0, 1],
                ret_bbox[2, 0],
                ret_bbox[2, 1],
            ]  # 4,
            trajectory.bbox_lst.append(bbox)  # bbox
            trajectory.frame_rgb_lst.append(frame_rgb)

        global_bbox = average_bbox_lst(trajectory.bbox_lst)

        for idx, (frame_rgb, lmk) in enumerate(zip(trajectory.frame_rgb_lst, trajectory.lmk_lst)):
            ret_dct = crop_image_by_bbox(
                frame_rgb,
                global_bbox,
                lmk=lmk,
                dsize=kwargs.get("dsize", 512),
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop"])

        if not trajectory.frame_rgb_crop_lst or len(trajectory.frame_rgb_crop_lst) == 0:
            return None

        if not trajectory.lmk_lst or len(trajectory.lmk_lst) == 0:
            return None

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
        }


    def calc_lmks_from_cropped_video(self, driving_rgb_crop_lst, crop_cfg: CropConfig, **kwargs):
        """Tracking based landmarks/alignment"""
        trajectory = Trajectory()
        idx_detect = kwargs.get("idx", 0)

        for idx, frame_rgb_crop in enumerate(driving_rgb_crop_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb_crop[..., ::-1]),  # convert to BGR
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                    crop=crop_cfg.crop,
                    id=idx_detect
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    raise Exception(f"No face detected in the frame #{idx}")
                elif len(src_face) > 1:
                    print(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {crop_cfg.direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb_crop, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb_crop, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)

        if not trajectory.lmk_lst or len(trajectory.lmk_lst) == 0:
            return None
        return trajectory.lmk_lst

    def crop_source_video(self, source_rgb_lst, crop_cfg: CropConfig, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        idx_detect = kwargs.get("idx", 0)

        for idx, frame_rgb in enumerate(source_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                    crop=crop_cfg.crop,
                    id=idx_detect
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    print(f"More than one face detected in the source frame_{idx}, only pick one face by rule {crop_cfg.direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)

            # crop the face
            ret_dct = crop_image(
                frame_rgb,  # ndarray
                lmk,  # 106x2 or Nx2
                dsize=crop_cfg.dsize,
                scale=crop_cfg.scale,
                vx_ratio=crop_cfg.vx_ratio,
                vy_ratio=crop_cfg.vy_ratio,
                flag_do_rot=crop_cfg.flag_do_rot,
            )
            lmk = self.human_landmark_runner.run(frame_rgb, lmk)
            ret_dct["lmk_crop"] = lmk

            # update a 256x256 version for network input
            ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize

            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_256x256"])
            trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        if not trajectory.frame_rgb_crop_lst or len(trajectory.frame_rgb_crop_lst) == 0:
            return None
        if not trajectory.lmk_crop_lst or len(trajectory.lmk_crop_lst) == 0:
            return None
        if not trajectory.M_c2o_lst or len(trajectory.M_c2o_lst) == 0:
            return None

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
            "M_c2o_lst": trajectory.M_c2o_lst,
        }