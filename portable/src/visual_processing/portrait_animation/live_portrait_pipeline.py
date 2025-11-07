# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import sys
import copy
import os.path as osp
from tqdm import tqdm
import imageio

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load, update_crop
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth, KalmanFilterLimit
from .utils.rprint import rlog as log
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    # Function for webcam working
    def execute_frame(self, frame, source_image_path):
        inference_cfg = self.live_portrait_wrapper.inference_cfg  # for convenience

        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
        log(f"Load source image from {source_image_path}")
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if inference_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)

        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if inference_cfg.flag_lip_zero:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, crop_info['lmk_crop'])
            if combined_lip_ratio_tensor_before_animation[0][0] < inference_cfg.lip_zero_threshold:
                inference_cfg.flag_lip_zero = False
            else:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        return x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb

    def generate_frame(self, x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, driving_info):
        inference_cfg = self.live_portrait_wrapper.inference_cfg  # for convenience

        # Process driving info
        driving_rgb = cv2.resize(driving_info, (256, 256))
        I_d_i = self.live_portrait_wrapper.prepare_videos([driving_rgb])[0]

        x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
        R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

        R_new = R_d_i @ R_s
        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_s_info['exp'])
        scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_s_info['scale'])
        t_new = x_s_info['t'] + (x_d_i_info['t'] - x_s_info['t'])
        t_new[..., 2].fill_(0)  # zero tz

        x_d_i_new = scale_new * (x_s @ R_new + delta_new) + t_new
        if inference_cfg.flag_lip_zero and lip_delta_before_animation is not None:
            x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)

        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

        if inference_cfg.flag_pasteback:
            mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            I_p_i_to_ori_blend = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
            return I_p_i_to_ori_blend
        else:
            return I_p_i

    def make_motion_value(self, I_i):
        x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
        R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

        item_dct = {
            'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
            'R': R_i.cpu().numpy().astype(np.float32),
            'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
            't': x_i_info['t'].cpu().numpy().astype(np.float32),
            'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
            'x_s': x_s.cpu().numpy().astype(np.float32),
        }

        return item_dct


    def safety_execute(self, args: ArgumentConfig, progress_callback=None, **kwargs):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        source_crop_cfg = copy.deepcopy(self.cropper.crop_cfg)
        driving_crop_cfg = copy.deepcopy(self.cropper.crop_cfg)

        source_crop = kwargs.get('source_crop', None)
        source_fps = kwargs.get('source_fps', None)
        source_fps = int(source_fps) if source_fps else None
        driving_crop = kwargs.get('driving_crop', None)
        driving_fps = int(kwargs.get('driving_fps', 25))
        scaled_lip_opening = int(kwargs.get('scaled_lip_opening', 1.8))

        source_start = int(kwargs.get('source_start', 0) * source_fps) if source_fps is not None else 0
        source_end = int(kwargs.get('source_end', len(args.source)) * source_fps) if source_fps is not None else 0

        ######## load source input ########
        if not isinstance(args.source, list):
            raise Exception(f"Source has to be list of frame files path!")
        if len(args.source) == 1:
            print(f"Use source image")
            flag_is_source_video = False
        else:
            print(f"Use source video")
            flag_is_source_video = True

        ######## process driving info ########
        if not isinstance(args.driving, list):
            raise Exception(f"Driving has to be list of frame files path!")

        if len(args.driving) == 1:
            print(f"Use driving image")
            flag_is_driving_video = False
        else:
            print(f"Use driving video")
            flag_is_driving_video = True

        ######## update crop ########
        if source_crop is not None:
            source_crop_cfg.direction = 'distance-from-embedding' if flag_is_source_video else 'distance-from-retarget-face'
        if driving_crop is not None:
            driving_crop_cfg.direction = 'distance-from-embedding' if flag_is_driving_video else 'distance-from-retarget-face'

        ######## make motion template ########
        print("Start making driving motion template...")

        video_format = kwargs.get('format', 'mp4')  # default is mp4 format
        codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
        quality = kwargs.get('quality')  # video quality
        pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
        macro_block_size = kwargs.get('macro_block_size', 2)
        ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

        wfp = osp.join(args.output_dir, f'retargeting.mp4')
        writer = imageio.get_writer(
            wfp, fps=driving_fps if source_fps is None else source_fps, format=video_format,
            codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat,
            macro_block_size=macro_block_size
        )

        source_n_frames = len(args.source)
        driving_n_frames = len(args.driving)
        n_frames = max(source_n_frames - source_start, driving_n_frames)

        driving_motion_zero = None

        source_motion_zero = None
        source_motion = None
        source_c_eyes = None

        # Smooth
        x_d_exp_kalman_filter = KalmanFilterLimit(process_variance = inf_cfg.driving_smooth_observation_variance)
        x_d_r_kalman_filter = KalmanFilterLimit(process_variance = inf_cfg.driving_smooth_observation_variance)

        ######## prepare for pasteback ########
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_relative_motion = inf_cfg.flag_relative_motion
        flag_normalize_lip, flag_relative_motion = True, True
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None

        # Clear prev face detection
        self.cropper.clear_face()

        if source_start != 0:
            for i in tqdm(range(source_start), desc="Extract source...", total=source_start, unit='iB', unit_scale=True, file=sys.stdout):
                # Source
                img_rgb = load_image_rgb(args.source[i])  # if frame last, when repeat
                source_img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
                writer.append_data(source_img_rgb)

        for i in tqdm(range(n_frames), desc="Animate...", total=n_frames, unit='iB', unit_scale=True, file=sys.stdout):
            # Source
            img_rgb = load_image_rgb(args.source[min(source_start + i, len(args.source) - 1)])  # if frame last, when repeat
            source_img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)

            if source_end < i and source_end != 0:
                # User cut start and end
                # Append source frame without change
                writer.append_data(source_img_rgb)
                continue

            if i == 0 and source_crop:
                print("Scale source crop")
                source_crop = update_crop(source_img_rgb, source_crop)
                source_crop_cfg.crop = source_crop

            if i == 0:
                # Check if the image contains a single, centered face
                is_valid_img = self.cropper.is_single_centered_face(source_img_rgb)
                # If the image is square and has one centered face, cropping is not needed (set flag_do_crop to False).
                # Otherwise, enable cropping (set flag_do_crop to True).
                inf_cfg.flag_do_crop = False if is_valid_img and source_img_rgb.shape[0] == source_img_rgb.shape[1] else True

            # Driving
            if flag_is_driving_video:
                if i >= len(args.driving):
                    # Append source frame without change
                    writer.append_data(source_img_rgb)
                    continue
                driving_img_rgb = load_image_rgb(args.driving[i])
            else:
                driving_img_rgb = load_image_rgb(args.driving[0])

            if not inf_cfg.flag_do_crop:
                # Resize driving
                driving_img_rgb = cv2.resize(driving_img_rgb, (source_img_rgb.shape[0], source_img_rgb.shape[1]))

            if i == 0 and driving_crop:
                print("Scale driving crop")
                driving_crop = update_crop(driving_img_rgb, driving_crop)
                driving_crop_cfg.crop = driving_crop

            # Detect face
            if inf_cfg.flag_crop_driving_video or (not is_square_video(driving_img_rgb)):
                ret_d = self.cropper.crop_driving_video([driving_img_rgb], crop_cfg=driving_crop_cfg, idx=3)
                if ret_d is None:
                    # Append source frame without change
                    writer.append_data(source_img_rgb)
                    continue
                print(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
            else:
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video([driving_img_rgb], crop_cfg=driving_crop_cfg, idx=3)
                if driving_lmk_crop_lst is None:
                    # Append source frame without change
                    writer.append_data(source_img_rgb)
                    continue
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in [driving_img_rgb]]

            driving_lmk_crop = driving_lmk_crop_lst[0]
            driving_rgb_crop_256x256 = driving_rgb_crop_256x256_lst[0]

            # calc eyes
            driving_c_d_eyes = calc_eye_close_ratio(driving_lmk_crop[None])
            driving_c_d_eyes = driving_c_d_eyes.astype(np.float32)
            driving_c_d_eyes = float(driving_c_d_eyes.mean())

            # calc lips
            driving_c_d_lip = calc_lip_close_ratio(driving_lmk_crop[None])
            driving_c_d_lip = driving_c_d_lip.astype(np.float32)
            driving_c_d_lip = float(driving_c_d_lip[0][0])

            # get I_d
            I_i = self.live_portrait_wrapper.prepare_source(driving_rgb_crop_256x256)

            # collect s, R, δ and t for inference
            driving_item_dct = self.make_motion_value(I_i)

            if i == 0:
                driving_motion_zero = driving_item_dct.copy()
            driving_motion = driving_item_dct
            driving_c_eyes = driving_c_d_eyes
            driving_c_lip = driving_c_d_lip

            # Detect face
            if inf_cfg.flag_do_crop:
                video_crop_info = self.cropper.crop_source_video([source_img_rgb], source_crop_cfg, idx=2)
            else:
                video_crop_info = self.cropper.calc_lmks_from_cropped_video([source_img_rgb], source_crop_cfg, idx=2)
            if video_crop_info is None:
                writer.append_data(source_img_rgb)
                continue
            if inf_cfg.flag_do_crop:
                img_crop_256x256, source_lmk_crop, source_M_c2o = video_crop_info['frame_crop_lst'][0], video_crop_info['lmk_crop_lst'][0], video_crop_info['M_c2o_lst'][0]
            else:
                img_crop_256x256, source_lmk_crop, source_M_c2o = cv2.resize(source_img_rgb, (256, 256)), video_crop_info[0], None
            source_lmk = source_lmk_crop

            # calc eyes
            source_c_d_eyes = calc_eye_close_ratio(source_lmk_crop[None])
            source_c_d_eyes = source_c_d_eyes.astype(np.float32)
            source_c_d_eyes = float(source_c_d_eyes.mean())

            # save the motion template
            # get I_d
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)

            if flag_is_source_video:
                # collect s, R, δ and t for inference
                source_item_dct = self.make_motion_value(I_s)

                if i == 0:
                    source_motion_zero = source_item_dct.copy()
                source_motion = source_item_dct
                source_c_eyes = source_c_d_eyes

                key_r = 'R' if 'R' in driving_motion_zero.keys() else 'R_d'  # compatible with previous keys
                if flag_relative_motion:
                    if flag_is_driving_video:
                        x_d_exp = source_motion['exp'] + driving_motion['exp'] - driving_motion_zero['exp']
                        x_d_exp_smooth = x_d_exp_kalman_filter.update(x_d_exp, shape=source_motion_zero['exp'].shape, device=device)
                    else:
                        x_d_exp = source_motion['exp'] + driving_motion_zero['exp'] - inf_cfg.lip_array
                        x_d_exp_smooth = torch.tensor(x_d_exp[0], dtype=torch.float32, device=device)
                    if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                        if flag_is_driving_video:
                            x_d_r = (np.dot(driving_motion[key_r], driving_motion_zero[key_r].transpose(0, 2, 1))) @ source_motion['R']
                            x_d_r_smooth = x_d_r_kalman_filter.update(x_d_r, shape=source_motion_zero['R'].shape, device=device)
                        else:
                            x_d_r = source_motion['R']
                            x_d_r_smooth = torch.tensor(x_d_r[0], dtype=torch.float32, device=device)
                else:
                    if flag_is_driving_video:
                        x_d_exp = driving_motion['exp']
                        x_d_exp_smooth = x_d_exp_kalman_filter.update(x_d_exp, shape=source_motion_zero['exp'].shape, device=device)
                    else:
                        x_d_exp = driving_motion_zero['exp']
                        x_d_exp_smooth = torch.tensor(x_d_exp[0], dtype=torch.float32, device=device)
                    if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                        if flag_is_driving_video:
                            x_d_r = driving_motion[key_r]
                            x_d_r_smooth = x_d_r_kalman_filter.update(x_d_r, shape=source_motion_zero['R'].shape, device=device)
                        else:
                            x_d_r = driving_motion_zero[key_r]
                            x_d_r_smooth = torch.tensor(x_d_r[0], dtype=torch.float32, device=device)
            else:
                x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
                x_c_s = x_s_info['kp']
                R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
                f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
                x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

                # let lip-open scalar to be 0 at first
                if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
                    c_d_lip_before_animation = [0.]
                    combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                    if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                        lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                # prepare paste back
                if source_M_c2o is not None:
                    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_M_c2o, dsize=(source_img_rgb.shape[1], source_img_rgb.shape[0]))

            ######## animate ########
            if i == 0:
                if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
                    print(f"The animated video consists of {n_frames} frames.")
                else:
                    print(f"The output of image-driven portrait animation is an image.")

            if flag_is_source_video:  # source video
                x_s_info = source_motion
                x_s_info = dct2device(x_s_info, device)

                f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)

                x_c_s = x_s_info['kp']
                R_s = x_s_info['R']
                x_s = x_s_info['x_s']

                # let lip-open scalar to be 0 at first if the input is a video
                if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
                    c_d_lip_before_animation = [0.]
                    combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                    if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                        lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                    else:
                        lip_delta_before_animation = None

                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if flag_source_video_eye_retargeting and source_lmk is not None:
                    if i == 0:
                        combined_eye_ratio_tensor_frame_zero = source_c_eyes
                        c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                        if c_d_eye_before_animation_frame_zero[0][0] < inf_cfg.source_video_eye_retargeting_threshold:
                            c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)
                # prepare paste back
                if source_M_c2o is not None:
                    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_M_c2o, dsize=(source_img_rgb.shape[1], source_img_rgb.shape[0]))

            if flag_is_source_video and not flag_is_driving_video:
                x_d_i_info = driving_motion_zero
            else:
                x_d_i_info = driving_motion
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()

            delta_new = x_s_info['exp'].clone()
            rounding = 1000

            if flag_relative_motion:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_smooth if flag_is_source_video else (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    if flag_is_source_video:
                        if flag_is_driving_video:  # Great part to smooth ending for driving to source
                            delta_new = x_s_info['exp'] + (1 - i / driving_n_frames) * (torch.floor(x_d_i_info['exp'] * rounding) / rounding - x_d_0_info['exp'])
                        else:
                            delta_new = x_s_info['exp'] + (1 - i / driving_n_frames) * (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
                    else:
                        if flag_is_driving_video:
                            delta_new = x_s_info['exp'] + (torch.floor(x_d_i_info['exp'] * rounding) / rounding - x_d_0_info['exp'])
                        else:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if flag_is_driving_video:
                            delta_new[:, lip_idx, :] = x_s_info['exp'][:, lip_idx, :] + ((torch.floor(x_d_i_info['exp'] * rounding) / rounding * scaled_lip_opening - x_d_0_info['exp'])[:,lip_idx, :])
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if flag_is_driving_video:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (torch.floor(x_d_i_info['exp'] * rounding) / rounding * scaled_lip_opening - x_d_0_info['exp']))[:, lip_idx, :]
                        else:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:,lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        if flag_is_driving_video:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (torch.floor(x_d_i_info['exp'] * rounding) / rounding - x_d_0_info['exp']))[:, eyes_idx, :]
                        else:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
                # if inf_cfg.animation_region == "all":
                #     scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                # else:
                #     scale_new = x_s_info['scale']
                scale_new = x_s_info['scale']  # not use x,y,z
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_smooth if flag_is_source_video else R_d_i
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    for idx in [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                        delta_new[:, idx, :] = x_d_exp_smooth[idx, :] if flag_is_source_video else x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_smooth[3:5, 1] if flag_is_source_video else x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_smooth[5, 2] if flag_is_source_video else x_d_i_info['exp'][:,5, 2]
                    delta_new[:, 8, 2] = x_d_exp_smooth[8, 2] if flag_is_source_video else x_d_i_info['exp'][:,8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_smooth[9, 1:] if flag_is_source_video else x_d_i_info['exp'][:,9,1:]
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_smooth[lip_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_smooth[eyes_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if flag_relative_motion and inf_cfg.driving_option == "expression-friendly" and not flag_is_source_video and flag_is_driving_video:
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                    # motion_multiplier *= inf_cfg.driving_multiplier
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                    c_d_eyes_i = driving_c_eyes
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                    c_d_lip_i = driving_c_lip
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

            # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
            I_p_pstbk = paste_back(I_p_i, source_M_c2o, source_img_rgb, mask_ori_float) if source_M_c2o is not None else I_p_i
            # Resize the result to match the size of source_img_rgb
            I_p_pstbk = cv2.resize(I_p_pstbk, (source_img_rgb.shape[1], source_img_rgb.shape[0]))
            writer.append_data(I_p_pstbk)

            # Updating progress
            if progress_callback:
                progress_callback(round(i / n_frames * 100, 0), "Animate...")

        inf_cfg.flag_do_crop = True  # set default value

        return wfp
