import os
import sys
import cv2
import uuid
import math
import numpy as np
import torch
from tqdm import tqdm
import onnxruntime

import insightface

from concurrent.futures import ThreadPoolExecutor
import threading

from visual_processing.face_detection.recognition import FaceRecognition
from .nudenet import NudeDetector


class FaceSwapProcessing:
    """
    Face swap by one photo
    """
    def __init__(self, model_path, face_swap_model_path, similarface = False, device="cpu", progress_callback=None):
        """
        Initialization
        :param model_path: path to model deepfake where will be download face recognition
        :param face_swap_model_path: path to face swap model
        """
        self.device = device

        # use cpu as with cuda on onnx can be problem
        self.access_providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in self.access_providers:
            provider = ["CUDAExecutionProvider"] if torch.cuda.is_available() and self.device != "cpu" else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]
        self.face_analysis_wrapper = FaceRecognition(
            name="buffalo_l",
            root=model_path,
            providers=provider,
        )
        self.face_analysis_wrapper.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.1)
        self.face_analysis_wrapper.warmup()

        self.face_swap_model = self.load(face_swap_model_path)
        self.filter_model = NudeDetector(providers=self.access_providers if self.device != "cpu" else None)
        self.face_target_fields = None
        self.similarface = similarface
        self.lock = threading.Lock()  # Create a lock
        self.progress = 0  # Initialize a progress counter
        self.progress_callback = progress_callback

    def load(self, face_swap_model_path):
        """
        Load model ONNX face swap
        :param face_swap_model_path: path to face swap model
        :return: loaded model
        """
        # use cpu as with cuda on onnx can be problem
        if "CUDAExecutionProvider" in self.access_providers:
            provider = ["CUDAExecutionProvider"] if self.device != "cpu" else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        return insightface.model_zoo.get_model(face_swap_model_path, providers=provider)

    def face_detect_with_alignment_from_source_frame(self, frame, face_fields=None):
        direction = 'distance-from-retarget-face' if face_fields is not None else 'left-right'
        dets = self.face_analysis_wrapper.get(frame, direction=direction, crop=face_fields)
        if not dets:
            raise FaceNotDetectedError("Face is not detected!")
        return dets[0]

    def face_detect_with_alignment_crop(self, image_files, face_fields):
        """
        Detect faces to swap if face_fields
        :param image_files: list of file paths of target images
        :param face_fields: crop target face
        :return:
        """
        predictions = []

        for n, image_file in enumerate(tqdm(image_files)):
            img_rgb = cv2.imread(image_file)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            dets = self.face_analysis_wrapper.get(
                img_bgr,
                flag_do_landmark_2d_106=True,
                direction='distance-from-embedding',
                max_face_num=0,
                crop=face_fields
            )
            if not dets:
                predictions.append([None])
                continue
            predictions.append([dets[0]])

            # Updating progress
            if self.progress_callback:
                self.progress_callback(round(n / len(image_files) * 100), "Face detection...")

        return predictions

    def face_detect_with_alignment_all(self, image_files):
        """
        Detect all faces in each image
        :param image_files: list of file paths of images
        :return: list of detected faces for each image
        """
        predictions = []
        for n, image_file in enumerate(tqdm(image_files)):
            img_bgr = cv2.imread(image_file)
            dets = self.face_analysis_wrapper.get(img_bgr, direction='large-small')
            if not dets:
                predictions.append([None])
                continue
            predictions.append(dets)
            # Updating progress
            if self.progress_callback:
                self.progress_callback(round(n / len(image_files) * 100, 0), "Face detection...")
        return predictions

    def process_frame(self, args):
        frame, face_det_result, source_face, progress_bar, output_directory, idx, num_frames = args
        tmp_frame = frame.copy()
        if self.filter_model.status(tmp_frame):
            for face in face_det_result:
                if face is None:
                    break
                else:
                    tmp_frame = self.face_swap_model.get(tmp_frame, face, source_face, paste_back=True)
        else:
            print("Face swap cannot be applied to nude or explicit content. Please upload images with appropriate content.")
        # Save the frame with zero-padded filename
        filename = os.path.join(output_directory, f"frame_{idx:04d}.png")
        cv2.imwrite(filename, tmp_frame)
        with self.lock:
            self.progress += 1
            progress_bar.update(1)  # Update progress bar in a thread-safe manner
            # Updating progress
            if self.progress_callback:
                self.progress_callback(round(self.progress / num_frames * 100), "Face swap thread...")
        return filename

    def swap_video(self, target_frames_path, source_face, target_face_fields, save_file: str, multiface=False, fps=30, start_frame=None, end_frame=None, video_format=".mp4"):
        if "CUDAExecutionProvider" in self.access_providers and torch.cuda.is_available() and self.device != "cpu":
            # thread will not work correct with GPU
            return self.swap_video_cuda(target_frames_path, source_face, target_face_fields, save_file, multiface, fps, start_frame, end_frame, video_format)
        else:
            return self.swap_video_thread(target_frames_path, source_face, target_face_fields, save_file, multiface, fps, start_frame, end_frame, video_format)

    def swap_video_thread(self, target_frames_path, source_face, target_face_fields, save_path: str, multiface=False, fps=30, start_frame=None, end_frame=None, video_format=".mp4"):
        """
        Face swap video with Threads. Will not work with CUDA
        :param target_frames_path: directory containing target frames
        :param source_face: source face
        :param target_face_fields: crop field for target face
        :param save_file: save file path
        :param multiface: bool use swap all face or use target crop
        :param fps: video fps
        :param video_format: video format
        :return:
        """
        file_name = str(uuid.uuid4()) + '.mp4'
        save_file = os.path.join(save_path, file_name)
        frame_save_path = os.path.join(save_path, "faceswap_frames")
        os.makedirs(frame_save_path, exist_ok=True)

        # Get a list of all frame files in the target_frames_path directory
        frame_files = sorted([os.path.join(target_frames_path, fname) for fname in os.listdir(target_frames_path) if fname.endswith('.jpg') or fname.endswith('.png')])
        full_frame_files = frame_files.copy()

        if start_frame and end_frame:
            frame_files = frame_files[start_frame:end_frame]

        # Read the first frame to get the dimensions
        first_frame = cv2.imread(frame_files[0])
        frame_h, frame_w = first_frame.shape[:-1]

        if video_format == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif video_format == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise ValueError("Unsupported video format: {}".format(video_format))

        out = cv2.VideoWriter(save_file, fourcc, fps, (frame_w, frame_h))

        # Adjust the way we get face detection results
        if multiface:
            print("Getting all face...")
            face_det_results = self.face_detect_with_alignment_all(frame_files)
        else:
            print("Getting target face...")
            face_det_results = self.face_detect_with_alignment_crop(frame_files, target_face_fields)

        print("Starting face swap...")

        progress_bar = tqdm(total=len(frame_files), unit='it', unit_scale=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            processed_frame_files = list(executor.map(self.process_frame, [
                (cv2.imread(frame_files[i]), face_det_results[i], source_face, progress_bar, frame_save_path, i, len(frame_files)) for i
                in range(len(frame_files))
            ]))

        progress_bar.close()

        processed_frame_files.sort()  # Ensure files are in the correct order

        if start_frame:
            for frame_file in full_frame_files[:start_frame]:
                frame = cv2.imread(frame_file)
                out.write(frame)

        for frame_file in processed_frame_files:
            frame = cv2.imread(frame_file)
            out.write(frame)

        if end_frame:
            for frame_file in full_frame_files[end_frame:]:
                frame = cv2.imread(frame_file)
                out.write(frame)

        out.release()
        print("Face swap processing finished...")
        return file_name

    def swap_video_cuda(self, target_frames_path, source_face, target_face_fields, save_path: str, multiface=False, fps=30, start_frame=None, end_frame=None, video_format=".mp4"):
        """Face swap video without Threads"""
        file_name = str(uuid.uuid4()) + '.mp4'
        save_file = os.path.join(save_path, file_name)

        # Get a list of all frame files in the target_frames_path directory
        frame_files = sorted([os.path.join(target_frames_path, fname) for fname in os.listdir(target_frames_path) if fname.endswith('.png')])
        full_frame_files = frame_files.copy()

        if start_frame and end_frame:
            frame_files = frame_files[start_frame:end_frame]

        # Read the first frame to get the dimensions
        first_frame = cv2.imread(frame_files[0])
        frame_h, frame_w = first_frame.shape[:-1]

        if video_format == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif video_format == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise ValueError("Unsupported video format: {}".format(video_format))

        out = cv2.VideoWriter(save_file, fourcc, fps, (frame_w, frame_h))

        # Adjust the way we get face detection results
        if multiface:
            print("Getting all face...")
            face_det_results = self.face_detect_with_alignment_all(frame_files)
        else:
            print("Getting target face...")
            face_det_results = self.face_detect_with_alignment_crop(frame_files, target_face_fields)

        print("Starting face swap...")
        progress_bar = tqdm(total=len(full_frame_files), unit='it', unit_scale=True)

        if start_frame:
            for i, frame_path in enumerate(full_frame_files[:start_frame]):
                tmp_frame = cv2.imread(frame_path)
                out.write(tmp_frame)
                progress_bar.update(1)

                # Updating progress
                if self.progress_callback:
                    self.progress_callback(round(i / len(full_frame_files[:start_frame]) * 100), "Update frame...")

        for i, dets in enumerate(face_det_results):
            tmp_frame = cv2.imread(frame_files[i])
            if self.filter_model.status(tmp_frame):
                for face in dets:
                    if face is None:
                        break
                    else:
                        tmp_frame = self.face_swap_model.get(tmp_frame, face, source_face, paste_back=True)
            else:
                print("Face swap cannot be applied to nude or explicit content. Please upload images with appropriate content.")
            out.write(tmp_frame)
            progress_bar.update(1)

            # Updating progress
            if self.progress_callback:
                self.progress_callback(round(i / len(face_det_results) * 100), "Face swap...")

        if end_frame:
            for i, frame_path in enumerate(full_frame_files[end_frame:]):
                tmp_frame = cv2.imread(frame_path)
                out.write(tmp_frame)
                progress_bar.update(1)

                # Updating progress
                if self.progress_callback:
                    self.progress_callback(round(i / len(full_frame_files[end_frame:]) * 100), "Update frame...")

        out.release()
        print("Face swap processing finished...")
        progress_bar.close()

        return file_name

    @staticmethod
    def get_center(crop: dict = None):
        """
        Get center coord
        :return:
        """
        if crop is not None:
            return int(crop["x"] + crop["width"] / 2), int(crop["y"] + crop["height"] / 2)
        return None, None

    def swap_image(self, target_frame, source_face, face_fields, save_dir: str, multiface=False):
        save_file = os.path.join(save_dir, "swapped_image.png")
        if self.filter_model.status(target_frame):
            x_center, y_center = self.get_center(face_fields)
            direction = 'distance-from-retarget-face' if face_fields is not None else 'left-right'
            dets = self.face_analysis_wrapper.get(target_frame, direction=direction, crop=face_fields)
            if not dets:
                raise FaceNotDetectedError("Face is not detected in target image!")

            if x_center is None or y_center is None or multiface:
                for face in dets:
                    target_frame = self.face_swap_model.get(target_frame, face, source_face, paste_back=True)
                else:
                    cv2.imwrite(save_file, target_frame)
                    return target_frame
            else:
                target_frame = self.face_swap_model.get(target_frame, dets[0], source_face, paste_back=True)
                cv2.imwrite(save_file, target_frame)
                return target_frame
        else:
            print("Face swap cannot be applied to nude or explicit content. Please upload images with appropriate content.")
            cv2.imwrite(save_file, target_frame)
            return target_frame


class FaceNotDetectedError(Exception):
    pass
