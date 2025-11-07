import os
import os.path as osp
import sys
import subprocess
import pickle as pkl
from typing import Literal
import torch
from tqdm import tqdm
import numpy as np
import uuid
import cv2
import imageio
from typing import List, Tuple, Union

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .live_portrait_pipeline import LivePortraitPipeline

from .utils.io import load_img_online, load_video, resize_to_limit, update_crop
from .utils.video import get_fps, images2video
from .utils.filter import KalmanFilterLimit
from .utils.cropper import Cropper
from .utils.crop import paste_back, prepare_paste_back
from .utils.camera import get_rotation_matrix
from .utils.helper import mkdir
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio


class LivePortrait:
    def __init__(self, checkpoint_F: str, checkpoint_M: str, checkpoint_G: str, checkpoint_W: str, checkpoint_S: str,
                 insightface_root: str, landmark_ckpt_path: str, device: str = None, source_max_dim: int = None):

        self.args = ArgumentConfig(
            flag_force_cpu=True if device == "cpu" else False
        )

        # specify configs for inference
        self.inference_cfg = self.partial_fields(InferenceConfig, self.args.__dict__)
        self.inference_cfg.checkpoint_F = checkpoint_F  # path to checkpoint of F
        self.inference_cfg.checkpoint_M = checkpoint_M  # path to checkpoint pf M
        self.inference_cfg.checkpoint_G = checkpoint_G  # path to checkpoint of G
        self.inference_cfg.checkpoint_W = checkpoint_W  # path to checkpoint of W
        self.inference_cfg.checkpoint_S = checkpoint_S  # path to checkpoint to S and R_eyes, R_lip
        self.inference_cfg.source_max_dim = source_max_dim if source_max_dim is None else self.inference_cfg.source_max_dim

        self.crop_cfg = self.partial_fields(CropConfig, self.args.__dict__)
        self.crop_cfg.insightface_root = insightface_root
        self.crop_cfg.landmark_ckpt_path = landmark_ckpt_path

        self.live_portrait_pipeline = self.load_models()

    def load_models(self):
        return LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg
        )

    @staticmethod
    def update_args(args, user_args):
        """update the args according to user inputs
        """
        for k, v in user_args.items():
            if hasattr(args, k):
                setattr(args, k, v)
        return args

    @torch.no_grad()
    def update_delta_new_eyeball_direction(self, eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
        if eyeball_direction_x > 0:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
            delta_new[0, 15, 0] += eyeball_direction_x * 0.001
        else:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.001
            delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

        delta_new[0, 11, 1] += eyeball_direction_y * -0.001
        delta_new[0, 15, 1] += eyeball_direction_y * -0.001
        blink = -eyeball_direction_y / 2.

        delta_new[0, 11, 1] += blink * -0.001
        delta_new[0, 13, 1] += blink * 0.0003
        delta_new[0, 15, 1] += blink * -0.001
        delta_new[0, 16, 1] += blink * 0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_smile(self, smile, delta_new, **kwargs):

        # Let's try to reverse-engineer the delta_new values
        delta_new[0, 20, 1] += smile * -0.01
        delta_new[0, 14, 1] += smile * -0.02
        delta_new[0, 17, 1] += smile * 0.0065
        delta_new[0, 17, 2] += smile * 0.003
        delta_new[0, 13, 1] += smile * -0.00275
        delta_new[0, 16, 1] += smile * -0.00275
        delta_new[0, 3, 1] += smile * -0.0035
        delta_new[0, 7, 1] += smile * -0.0035

        return delta_new

    @torch.no_grad()
    def update_delta_new_wink(self, wink, delta_new, **kwargs):

        delta_new[0, 11, 1] += wink * 0.001
        delta_new[0, 13, 1] += wink * -0.0003
        delta_new[0, 17, 0] += wink * 0.0003
        delta_new[0, 17, 1] += wink * 0.0003
        delta_new[0, 3, 1] += wink * -0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_eyebrow(self, eyebrow, delta_new, **kwargs):
        if eyebrow > 0:
            delta_new[0, 1, 1] += eyebrow * 0.001
            delta_new[0, 2, 1] += eyebrow * -0.001
        else:
            delta_new[0, 1, 0] += eyebrow * -0.001
            delta_new[0, 2, 0] += eyebrow * 0.001
            delta_new[0, 1, 1] += eyebrow * 0.0003
            delta_new[0, 2, 1] += eyebrow * -0.0003
        return delta_new

    @torch.no_grad()
    def update_delta_new_pouting(self, lip_variation_zero, delta_new, **kwargs):
        delta_new[0, 19, 0] += lip_variation_zero

        return delta_new

    @torch.no_grad()
    def update_delta_new_pursing(self, lip_variation_one, delta_new, **kwargs):
        delta_new[0, 14, 1] += lip_variation_one * 0.001
        delta_new[0, 3, 1] += lip_variation_one * -0.0005
        delta_new[0, 7, 1] += lip_variation_one * -0.0005
        delta_new[0, 17, 2] += lip_variation_one * -0.0005

        return delta_new

    @torch.no_grad()
    def update_delta_new_grin(self, lip_variation_two, delta_new, **kwargs):
        delta_new[0, 20, 2] += lip_variation_two * -0.001
        delta_new[0, 20, 1] += lip_variation_two * -0.001
        delta_new[0, 14, 1] += lip_variation_two * -0.001

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_expression(self, lip_variation_three, delta_new, **kwargs):
        delta_new[0, 19, 1] += lip_variation_three * 0.001
        delta_new[0, 19, 2] += lip_variation_three * 0.0001
        delta_new[0, 17, 1] += lip_variation_three * -0.0001

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_x(self, mov_x, delta_new, **kwargs):
        delta_new[0, 5, 0] += mov_x

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_y(self, mov_y, delta_new, **kwargs):
        delta_new[0, 5, 1] += mov_y

        return delta_new

    @staticmethod
    def update_lst(val_lst: List[Tuple[int, float]], n_frames: int) -> List[float]:
        # Sort the list of tuples based on the first element of each tuple
        val_lst = sorted(val_lst, key=lambda x: x[0])
        # Add the initial tuple (0, value) with the value from the first tuple in the sorted list
        first_val = (0, val_lst[0][1])
        val_lst = [first_val] + val_lst  # Insert (0, value) at the beginning
        # Remove duplicates while preserving order (not needed here if no duplicates)
        val_lst = list(dict.fromkeys(val_lst))

        # Initialize an empty list to store the expanded values
        expanded_vals = []
        # Iterate over the sorted list of tuples
        for i in range(len(val_lst) - 1):
            start_index, start_val = val_lst[i]
            end_index, end_val = val_lst[i + 1]
            # Determine the range to be filled
            for j in range(start_index, end_index):
                # Linear interpolation between start_val and end_val
                ratio = (j - start_index) / (end_index - start_index)
                interpolated_val = start_val + ratio * (end_val - start_val)
                expanded_vals.append(interpolated_val)

        # Fill the last segment with the last value
        last_index, last_val = val_lst[-1]
        for j in range(last_index, n_frames):
            expanded_vals.append(last_val)

        return expanded_vals

    @staticmethod
    def reset_face_analysis(pipeline):
        pipeline.cropper.clear_face()

    def extract_portrait_parameters(self, img_src, pipeline, crop: dict = None):
        img_rgb = load_img_online(img_src, mode='rgb', max_dim=self.inference_cfg.source_max_dim, n=2)

        if crop is not None:
            # print("Scale crop")
            crop = update_crop(img_rgb, crop)
            pipeline.cropper.crop_cfg.crop = crop
            pipeline.cropper.crop_cfg.direction = 'distance-from-retarget-face'
        image_crop_info = pipeline.cropper.crop_source_image(img_rgb, pipeline.cropper.crop_cfg)
        if image_crop_info is None:
            print("image_crop_info is None")
            return 0, 0, 0, 0, 0
        source_eye_ratio = calc_eye_close_ratio(image_crop_info['lmk_crop'][None])
        source_lip_ratio = calc_lip_close_ratio(image_crop_info['lmk_crop'][None])
        # print("Calculating eyes-open and lip-open ratios successfully!")
        I_s = pipeline.live_portrait_wrapper.prepare_source(image_crop_info['img_crop_256x256'])
        # I_s = pipeline.live_portrait_wrapper.prepare_source(img_rgb)
        x_s_info = pipeline.live_portrait_wrapper.get_kp_info(I_s)
        return (
            round(float(source_eye_ratio.mean()), 2), round(float(source_lip_ratio[0][0]), 2),
            round(float(x_s_info['pitch'].item()), 4), round(float(x_s_info['yaw'].item()), 4),
            round(float(x_s_info['roll'].item()), 4)
        )

    def update_image_portrait_parameters(self, img_src, pipeline, eye_ratio: float = 0, lip_ratio: float = 0, pitch: float = 0,
                                        yaw: float = 0, roll: float = 0, x: float = 0, y: float = 0, z: float = 1,
                                        eyeball_direction_x: float = 0, eyeball_direction_y: float = 0,
                                        smile: float = 0, wink: float = 0, eyebrow: float = 0, grin: float = 0,
                                        pursing: float = 0, pouting: float = 0, lip_expression: float = 0, crop: dict = None):
        device = pipeline.live_portrait_wrapper.device
        inference_cfg = pipeline.live_portrait_wrapper.inference_cfg
        img_rgb = load_img_online(img_src, mode='rgb', max_dim=inference_cfg.source_max_dim, n=2)

        if crop is not None:
            # print("Scale crop")
            crop = update_crop(img_rgb, crop)
            pipeline.cropper.crop_cfg.crop = crop
            pipeline.cropper.crop_cfg.direction = 'distance-from-retarget-face'

        # Check if the image contains a single, centered face
        is_valid_img = pipeline.cropper.is_single_centered_face(img_rgb)
        # If the image is square and has one centered face, cropping is not needed (set flag_do_crop to False).
        # Otherwise, enable cropping (set flag_do_crop to True).
        self.inference_cfg.flag_do_crop = False if is_valid_img and img_rgb.shape[0] == img_rgb.shape[1] else True

        if self.inference_cfg.flag_do_crop:
            image_crop_info = pipeline.cropper.crop_source_image(img_rgb, pipeline.cropper.crop_cfg)
        else:
            image_crop_info = pipeline.cropper.calc_lmk_source_image(img_rgb, pipeline.cropper.crop_cfg)
        if image_crop_info is None:
            return img_rgb  # Return source image

        if self.inference_cfg.flag_do_crop:
            img_crop_256x256, source_lmk, source_M_c2o = image_crop_info['img_crop_256x256'], image_crop_info['lmk_crop'], image_crop_info['M_c2o']
        else:
            img_crop_256x256, source_lmk, source_M_c2o = cv2.resize(img_rgb, (256, 256)), image_crop_info, None

        source_eye_ratio = calc_eye_close_ratio(source_lmk[None])
        source_lip_ratio = calc_lip_close_ratio(source_lmk[None])

        I_s = pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
        source_lmk_user = source_lmk
        crop_M_c2o = source_M_c2o
        mask_ori = prepare_paste_back(inference_cfg.mask_crop, source_M_c2o, dsize=(img_rgb.shape[1], img_rgb.shape[0])) if self.inference_cfg.flag_do_crop else None

        x_s_info = pipeline.live_portrait_wrapper.get_kp_info(I_s)

        pitch = torch.tensor([[pitch]]).to(device)
        yaw = torch.tensor([[yaw]]).to(device)
        roll = torch.tensor([[roll]]).to(device)

        R_s_user = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])  # Started params
        R_d_user = get_rotation_matrix(pitch, yaw, roll)  # Updated params

        f_s_user = pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s_user = pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

        x_s_user = x_s_user.to(device)
        f_s_user = f_s_user.to(device)
        R_s_user = R_s_user.to(device)
        R_d_user = R_d_user.to(device)

        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        z = torch.tensor(z).to(device)

        eyeball_direction_x = torch.tensor(eyeball_direction_x).to(device)
        eyeball_direction_y = torch.tensor(eyeball_direction_y).to(device)

        smile = torch.tensor(smile).to(device)
        wink = torch.tensor(wink).to(device)
        eyebrow = torch.tensor(eyebrow).to(device)
        grin = torch.tensor(grin).to(device)
        pursing = torch.tensor(pursing).to(device)
        pouting = torch.tensor(pouting).to(device)
        lip_expression = torch.tensor(lip_expression).to(device)

        x_c_s = x_s_info['kp'].to(device)

        # This is the value we're looking for
        delta_new = x_s_info['exp'].to(device)

        scale_new = x_s_info['scale'].to(device)
        t_new = x_s_info['t'].to(device)
        R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

        if eyeball_direction_x != 0 or eyeball_direction_y != 0:
            delta_new = self.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
        if smile != 0:
            delta_new = self.update_delta_new_smile(smile, delta_new)
        if wink != 0:
            delta_new = self.update_delta_new_wink(wink, delta_new)
        if eyebrow != 0:
            delta_new = self.update_delta_new_eyebrow(eyebrow, delta_new)
        if grin != 0:
            delta_new = self.update_delta_new_grin(grin, delta_new)
        if pursing != 0:
            delta_new = self.update_delta_new_pursing(pursing, delta_new)
        if pouting != 0:
            delta_new = self.update_delta_new_pouting(pouting, delta_new)
        if lip_expression != 0:
            delta_new = self.update_delta_new_lip_expression(lip_expression, delta_new)
        if x != 0:
            delta_new = self.update_delta_new_mov_x(-x, delta_new)
        if y != 0:
            delta_new = self.update_delta_new_mov_y(y, delta_new)

        # Sum up delta_new, which contains the modifications.
        x_d_new = z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
        eyes_delta, lip_delta = None, None

        if eye_ratio != round(float(source_eye_ratio.mean()), 2):
            combined_eye_ratio_tensor = pipeline.live_portrait_wrapper.calc_combined_eye_ratio([[float(eye_ratio)]], source_lmk_user)
            eyes_delta = pipeline.live_portrait_wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor)
        if lip_ratio != round(float(source_lip_ratio[0][0]), 2):
            combined_lip_ratio_tensor = pipeline.live_portrait_wrapper.calc_combined_lip_ratio([[float(lip_ratio)]], source_lmk_user)
            lip_delta = pipeline.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor)

        # X_d_new
        x_d_new = x_d_new + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
        x_d_new = pipeline.live_portrait_wrapper.stitching(x_s_user, x_d_new)

        out = pipeline.live_portrait_wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
        out = pipeline.live_portrait_wrapper.parse_output(out['out'])[0]

        out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori) if mask_ori is not None else out
        return out_to_ori_blend

    def update_video_portrait_parameters(self, frames_files, pipeline, output: str, fps: int = 25, eye_ratio: list = None,
                                         lip_ratio: list = None, pitch: list = None, yaw: list = None, roll: list = None,
                                         x: list = None, y: list = None, z: list = None,
                                         eyeball_direction_x: list = None, eyeball_direction_y: list = None,
                                         smile: list = None, wink: list = None, eyebrow: list = None, grin: list = None,
                                         pursing: list = None, pouting: list = None, lip_expression: list = None,
                                         smooth_strength: float = 0.0003, crops: list = None, progress_callback=None, **kwargs):
        device = pipeline.live_portrait_wrapper.device
        inference_cfg = pipeline.live_portrait_wrapper.inference_cfg

        n_frames = len(frames_files)

        # Update params lists (Batch process updates as lists)
        eye_ratio = self.update_lst(eye_ratio, n_frames) if eye_ratio and isinstance(eye_ratio, list) else None
        lip_ratio = self.update_lst(lip_ratio, n_frames) if lip_ratio and isinstance(lip_ratio, list) else None
        pitch = self.update_lst(pitch, n_frames) if pitch and isinstance(pitch, list) else None
        yaw = self.update_lst(yaw, n_frames) if yaw and isinstance(yaw, list) else None
        roll = self.update_lst(roll, n_frames) if roll and isinstance(roll, list) else None

        x = self.update_lst(x if x is not None else [0] * n_frames, n_frames)
        y = self.update_lst(y if y is not None else [0] * n_frames, n_frames)
        z = self.update_lst(z if z is not None else [1] * n_frames, n_frames)
        eyeball_direction_x = self.update_lst(eyeball_direction_x if eyeball_direction_x is not None else [0] * n_frames, n_frames)
        eyeball_direction_y = self.update_lst( eyeball_direction_y if eyeball_direction_y is not None else [0] * n_frames, n_frames)
        smile = self.update_lst(smile if smile is not None else [0] * n_frames, n_frames)
        wink = self.update_lst(wink if wink is not None else [0] * n_frames, n_frames)
        eyebrow = self.update_lst(eyebrow if eyebrow is not None else [0] * n_frames, n_frames)
        grin = self.update_lst(grin if grin is not None else [0] * n_frames, n_frames)
        pursing = self.update_lst(pursing if pursing is not None else [0] * n_frames, n_frames)
        pouting = self.update_lst(pouting if pouting is not None else [0] * n_frames, n_frames)
        lip_expression = self.update_lst(lip_expression if lip_expression is not None else [0] * n_frames, n_frames)

        lip_kalman_filter = KalmanFilterLimit(smooth_strength)
        eye_kalman_filter = KalmanFilterLimit(smooth_strength)

        video_format = kwargs.get('format', 'mp4')  # default is mp4 format
        codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
        quality = kwargs.get('quality')  # video quality
        pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
        macro_block_size = kwargs.get('macro_block_size', 2)
        ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

        wfp = osp.join(output, f'{uuid.uuid4()}_retargeting.mp4')
        writer = imageio.get_writer(
            wfp, fps=fps, format=video_format,
            codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat,
            macro_block_size=macro_block_size
        )

        get_crop = lambda i, data: next((d for key, lst in data if key == i for d in lst), None)

        for i, img_src in tqdm(enumerate(frames_files), desc="Writing...", total=len(frames_files), unit='frames', file=sys.stdout):
            source_rgb = load_img_online(img_src, mode='rgb', max_dim=inference_cfg.source_max_dim, n=inference_cfg.source_division)
            # Get crop
            crop = get_crop(i, crops)
            if crop:
                # print("Scale crop")
                crop = update_crop(source_rgb, crop)
                pipeline.cropper.crop_cfg.crop = crop
                pipeline.cropper.crop_cfg.direction = 'distance-from-embedding'
                pipeline.cropper.update_center_face(crop)

            if i == 0:
                # Check if the image contains a single, centered face
                is_valid_img = pipeline.cropper.is_single_centered_face(source_rgb)
                # If the image is square and has one centered face, cropping is not needed (set flag_do_crop to False).
                # Otherwise, enable cropping (set flag_do_crop to True).
                self.inference_cfg.flag_do_crop = False if is_valid_img and source_rgb.shape[0] == source_rgb.shape[1] else True

            # Detect face
            if self.inference_cfg.flag_do_crop:
                image_crop_info = pipeline.cropper.crop_source_image(source_rgb, pipeline.cropper.crop_cfg)
            else:
                image_crop_info = pipeline.cropper.calc_lmk_source_image(source_rgb, pipeline.cropper.crop_cfg)
            if image_crop_info is None:
                writer.append_data(source_rgb)
                continue

            if self.inference_cfg.flag_do_crop:
                img_crop_256x256, source_lmk, source_M_c2o = image_crop_info['img_crop_256x256'], image_crop_info['lmk_crop_256x256'], image_crop_info['M_c2o']
                mask_ori_lst = prepare_paste_back(inference_cfg.mask_crop, source_M_c2o, dsize=(source_rgb.shape[1], source_rgb.shape[0]))
            else:
                img_crop_256x256, source_lmk, source_M_c2o = cv2.resize(source_rgb, (256, 256)), image_crop_info, None
                mask_ori_lst = None

            # save the motion template
            I_s = pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)

            # update eyes
            source_eye_ratio = calc_eye_close_ratio(source_lmk[None])
            source_eye_ratio = float(source_eye_ratio.mean())

            if eye_ratio is not None:
                source_eye_ratio += eye_ratio[i]
                source_eye_ratio = max(0.0, min(0.8, source_eye_ratio))  # valid value

            # update lips
            source_lip_ratio = calc_lip_close_ratio(source_lmk[None])
            source_lip_ratio = float(source_lip_ratio[0][0])

            if lip_ratio is not None:
                source_lip_ratio += lip_ratio[i]
                source_lip_ratio = max(0.0, min(0.8, source_lip_ratio))  # valid value

            # Put new x_s_info here (after I_s has been declared)
            x_s_info = pipeline.live_portrait_wrapper.get_kp_info(I_s)

            x_d_info_user_pitch = torch.tensor([[pitch[i] + float(x_s_info['pitch'].item())]]).to(device) if pitch is not None else x_s_info['pitch']
            x_d_info_user_yaw = torch.tensor([[yaw[i] + float(x_s_info['yaw'].item())]]).to(device) if yaw is not None else x_s_info['yaw']
            x_d_info_user_roll = torch.tensor([[roll[i] + float(x_s_info['roll'].item())]]).to(device) if roll is not None else x_s_info['roll']

            x_d_info_user_pitch = max(-90, min(90, x_d_info_user_pitch))  # valid value
            x_d_info_user_yaw = max(-90, min(90, x_d_info_user_yaw))  # valid value
            x_d_info_user_roll = max(-90, min(90, x_d_info_user_roll))  # valid value

            R_s_user = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)

            f_s_user = pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s_user = pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

            c_d_lip_retargeting = [source_lip_ratio]
            c_d_eye_retargeting = [[float(source_eye_ratio)]]

            combined_lip_ratio_tensor_retargeting = pipeline.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_retargeting, source_lmk)
            combined_eye_ratio_tensor_retargeting = pipeline.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eye_retargeting, source_lmk)

            lip_delta_retargeting = pipeline.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor_retargeting)
            eye_delta_retargeting = pipeline.live_portrait_wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor_retargeting)

            # Append to lists
            lip_delta_retargeting_cpu = lip_delta_retargeting.cpu().numpy().astype(np.float32)
            eye_delta_retargeting_cpu = eye_delta_retargeting.cpu().numpy().astype(np.float32)

            # Smooth lips and eyes
            lip_delta_retargeting_smooth = lip_kalman_filter.update(lip_delta_retargeting_cpu, shape=lip_delta_retargeting_cpu.shape, device=device)
            eye_delta_retargeting_smooth = eye_kalman_filter.update(eye_delta_retargeting_cpu, shape=eye_delta_retargeting_cpu.shape, device=device)

            x_user = torch.tensor(x[i]).to(device)
            y_user = torch.tensor(y[i]).to(device)
            z_user = torch.tensor(z[i]).to(device)

            eyeball_direction_x_user = torch.tensor(eyeball_direction_x[i]).to(device)
            eyeball_direction_y_user = torch.tensor(eyeball_direction_y[i]).to(device)

            smile_user = torch.tensor(smile[i]).to(device)
            wink_user = torch.tensor(wink[i]).to(device)
            eyebrow_user = torch.tensor(eyebrow[i]).to(device)
            grin_user = torch.tensor(grin[i]).to(device)
            pursing_user = torch.tensor(pursing[i]).to(device)
            pouting_user = torch.tensor(pouting[i]).to(device)

            # Retarget video in n frames
            x_s_user_i = x_s_user
            f_s_user_i = f_s_user
            R_s_user_i = R_s_user.to(device)
            R_d_user_i = R_d_user.to(device)

            x_s_info_i = x_s_info
            source_lmk_crop = source_lmk

            x_c_i_s = x_s_info_i['kp'].to(device)
            delta_i_new = x_s_info_i['exp'].to(device)
            scale_i_new = x_s_info_i['scale'].to(device)
            t_i_new = x_s_info_i['t'].to(device)
            R_d_i_new = (R_d_user_i @ R_s_user_i.permute(0, 2, 1)) @ R_s_user_i

            if eyeball_direction_x_user != 0 or eyeball_direction_y_user != 0:
                delta_i_new = self.update_delta_new_eyeball_direction(eyeball_direction_x_user, eyeball_direction_y_user, delta_i_new)
            if smile_user != 0:
                delta_i_new = self.update_delta_new_smile(smile_user, delta_i_new)
            if wink_user != 0:
                delta_i_new = self.update_delta_new_wink(wink_user, delta_i_new)
            if eyebrow_user != 0:
                delta_i_new = self.update_delta_new_eyebrow(eyebrow_user, delta_i_new)
            if grin_user != 0:
                delta_i_new = self.update_delta_new_grin(grin_user, delta_i_new)
            if pursing_user != 0:
                delta_i_new = self.update_delta_new_pursing(pursing_user, delta_i_new)
            if pouting_user != 0:
                delta_i_new = self.update_delta_new_pouting(pouting_user, delta_i_new)
            if lip_expression[i] != 0:
                delta_i_new = self.update_delta_new_lip_expression(lip_expression[i], delta_i_new)
            if x_user != 0:
                delta_i_new = self.update_delta_new_mov_x(-x_user, delta_i_new)
            if y_user != 0:
                delta_i_new = self.update_delta_new_mov_y(y_user, delta_i_new)

            x_d_i_new = z_user * scale_i_new * (x_c_i_s @ R_d_i_new + delta_i_new) + t_i_new
            eyes_i_delta, lip_i_delta = None, None

            if source_eye_ratio != 0:
                combined_eye_ratio_tensor = pipeline.live_portrait_wrapper.calc_combined_eye_ratio([[float(source_eye_ratio)]], source_lmk_crop)
                eyes_i_delta = pipeline.live_portrait_wrapper.retarget_eye(x_s_user_i, combined_eye_ratio_tensor)
            if source_lip_ratio != 0:
                combined_lip_ratio_tensor = pipeline.live_portrait_wrapper.calc_combined_lip_ratio([[float(source_lip_ratio)]], source_lmk_crop)
                lip_i_delta = pipeline.live_portrait_wrapper.retarget_lip(x_s_user_i, combined_lip_ratio_tensor)

            # Smoothing (only in video) TODO commented this or not?
            lip_delta_retargeting = lip_delta_retargeting_smooth
            eye_delta_retargeting = eye_delta_retargeting_smooth

            x_d_i_new2 = x_s_user_i + lip_delta_retargeting + eye_delta_retargeting

            # x_d_new behavior from execute_image_retargeting
            x_d_i_new = x_d_i_new + (eyes_i_delta if eyes_i_delta is not None else 0) + (lip_i_delta if lip_i_delta is not None else 0)

            # Stitching all x's
            x_d_i_new = pipeline.live_portrait_wrapper.stitching(x_s_user_i, x_d_i_new)
            x_d_i_new = pipeline.live_portrait_wrapper.stitching(x_d_i_new2, x_d_i_new)
            out = pipeline.live_portrait_wrapper.warp_decode(f_s_user_i, x_s_user_i, x_d_i_new)
            I_p_i = pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_pstbk = paste_back(I_p_i, source_M_c2o, source_rgb, mask_ori_lst) if mask_ori_lst is not None and source_M_c2o is not None else I_p_i

            if I_p_pstbk is not None:
                writer.append_data(I_p_pstbk)
            else:
                writer.append_data(I_p_i)

            # Updating progress
            if progress_callback:
                progress_callback(round((i + 1) / n_frames * 100, 0), "Retarget portrait...")

        writer.close()
        return wfp


    @staticmethod
    def make_abs_path(fn):
        return osp.join(osp.dirname(osp.realpath(__file__)), fn)

    @staticmethod
    def load_lip_array(model):
        with open(model, 'rb') as f:
            return pkl.load(f)

    @staticmethod
    def partial_fields(target_class, kwargs):
        return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

    @staticmethod
    def validate_audio_priority(priority: str) -> Union[str, Literal["source", "driving"]]:
        allowed_priority = ["source", "driving"]
        if priority in allowed_priority:
            return priority  # Return the region if it's valid
        else:
            return "driving"  # Default to "all" if invalid

    @staticmethod
    def validate_animation_region(region: str) -> Union[str, Literal["exp", "pose", "lip", "eyes", "all"]]:
        allowed_regions = ["exp", "pose", "lip", "eyes", "all"]
        if region in allowed_regions:
            return region  # Return the region if it's valid
        else:
            return "all"  # Default to "all" if invalid

    def animate(self, pipeline, source_files: list, source_fps: int, driving_files: list, driving_fps: int,
                output_dir: str, animation_region: str = None, source_crop: list = None,
                driving_crop: list = None, progress_callback=None, source_start: float = None,
                source_end: float = None, **kwargs):
        args = self.args
        args.source = source_files
        args.driving = driving_files
        args.output_dir = output_dir
        args.animation_region = self.validate_animation_region(animation_region)

        # the region where the animation was performed, "exp" means the expression, "pose" means the head pose
        pipeline.live_portrait_wrapper.inference_cfg.animation_region = self.validate_animation_region(animation_region)

        # run
        wfp = pipeline.safety_execute(args, source_crop=source_crop[0], source_fps=source_fps, driving_crop=driving_crop[0],
                                      driving_fps=driving_fps, progress_callback=progress_callback, source_start=source_start,
                                      source_end=source_end, **kwargs)
        return wfp
