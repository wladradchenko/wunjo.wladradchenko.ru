import os
import sys
import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm
import onnxruntime

if sys.platform == 'win32':
    from insightfacer.windows.model_zoo import get_model
elif sys.platform == 'darwin':
    from insightfacer.linux.model_zoo import get_model
elif sys.platform == 'linux':
    from insightfacer.linux.model_zoo import get_model

from concurrent.futures import ThreadPoolExecutor
import threading

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "deepfake"))
from src.face3d.recognition import FaceRecognition
sys.path.pop(0)


class FaceSwapDeepfake:
    """
    Face swap by one photo
    """
    def __init__(self, model_path, face_swap_model_path, similarface = False):
        """
        Initialization
        :param model_path: path to model deepfake where will be download face recognition
        :param face_swap_model_path: path to face swap model
        """
        self.face_recognition = FaceRecognition(model_path)
        self.access_providers = onnxruntime.get_available_providers()
        self.face_swap_model = self.load(face_swap_model_path)
        self.face_target_fields = None
        self.similarface = similarface
        self.lock = threading.Lock()  # Create a lock
        self.progress = 0  # Initialize a progress counter

    def load(self, face_swap_model_path):
        """
        Load model ONNX face swap
        :param face_swap_model_path: path to face swap model
        :return: loaded model
        """
        # use cpu as with cuda on onnx can be problem
        if "CUDAExecutionProvider" in self.access_providers:
            provider = ["CUDAExecutionProvider"] if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        return get_model(face_swap_model_path, providers=provider)

    @staticmethod
    def get_real_crop_box(frame, face_fields):
        """
        Get real crop box
        :param frame:
        :return:
        """
        if face_fields is not None:
            # user crop face in frontend
            squareFace = face_fields[0]
            # real size
            originalHeight, originalWidth, _ = frame.shape

            canvasWidth = squareFace["canvasWidth"]
            canvasHeight = squareFace["canvasHeight"]
            # Calculate the scale factor
            scaleFactorX = originalWidth / canvasWidth
            scaleFactorY = originalHeight / canvasHeight

            # Calculate the new position and size of the square face on the original image
            # Convert canvas square face coordinates to original image coordinates
            newX1 = squareFace['x'] * scaleFactorX
            newX2 = (squareFace['x'] + squareFace['width']) * scaleFactorX
            newY1 = squareFace['y'] * scaleFactorY
            newY2 = (squareFace['y'] + squareFace['height']) * scaleFactorY

            d_user_crop = dlib.rectangle(int(newX1), int(newY1), int(newX2), int(newY2))
            # get the center point of d_new
            user_crop_center = d_user_crop.center()

            return user_crop_center.x, user_crop_center.y
        return None, None

    def face_detect_with_alignment_from_source_frame(self, frame, face_fields=None):
        x_center, y_center = self.get_real_crop_box(frame, face_fields)
        dets = self.face_recognition.get_faces(frame)
        if not dets:
            raise FaceNotDetectedError("Face is not detected!")
        # it means what user will use multiface
        if x_center is None or y_center is None:
            return dets[0]
        else:
            for face in dets:
                x1, y1, x2, y2 = face.bbox
                if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                    return face

        raise FaceNotDetectedError("Face is not detected in user crop field!")

    def get_smoothened_boxes(self, boxes: np.ndarray, T: int = 5):
        """
        Get smoothened boxes
        :param boxes: frames with face boxes
        :param T: smooth windows size
        :return: list of processed frames, fps of the video
        """
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect_with_alignment_crop(self, images, face_fields):
        """
        Detect faces to swap if face_fields
        :param images: frames of target image
        :param face_fields: crop target face
        :return:
        """
        predictions = []
        face_embedding_list = []
        face_gender = None
        x_center, y_center = self.get_real_crop_box(images[0], face_fields)

        for image in tqdm(images):
            dets = self.face_recognition.get_faces(image)
            if not dets:
                predictions.append([None])
                continue
            # this is init first face
            if x_center is None or y_center is None:
                face = dets[0]  # get first face
                x1, y1, x2, y2 = face.bbox
                face_gender = face.gender  # face gender
                predictions.append([face])  # prediction
                face_embedding_list += [face.normed_embedding]
                x_center = int((x1 + x2) / 2)  # set new center
                y_center = int((y1 + y2) / 2)  # set new center
            elif not face_embedding_list:  # not face yet, set new face
                for face in dets:
                    x1, y1, x2, y2 = face.bbox
                    if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                        face_gender = face.gender  # face gender
                        predictions.append([face])  # prediction
                        face_embedding_list += [face.normed_embedding]
                        x_center = int((x1 + x2) / 2)  # set new center
                        y_center = int((y1 + y2) / 2)  # set new center
                        break
            else:  # here is already recognition
                local_face_param = []
                for i, face in enumerate(dets):
                    x1, y1, x2, y2 = face.bbox
                    x_center = int((x1 + x2) / 2)  # set new center
                    y_center = int((y1 + y2) / 2)  # set new center
                    normed_embedding = face.normed_embedding
                    is_similar = self.face_recognition.is_similar_face(normed_embedding, face_embedding_list)
                    if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                        local_face_param += [{
                            "is_center": True, "is_gender": face_gender == face.gender, "is_embed": is_similar,
                            "bbox": face.bbox, "gender": face.gender, "embed": normed_embedding, "id": i
                        }]
                    else:
                        local_face_param += [{
                            "is_center": False, "is_gender": face_gender == face.gender, "is_embed": is_similar,
                            "bbox": face.bbox, "gender": face.gender, "embed": normed_embedding, "id": i
                        }]
                local_predictions = []
                for param in local_face_param:
                    if (param["is_center"] or param["is_gender"]) and param["is_embed"]:
                        # this predicted
                        x1, y1, x2, y2 = param["bbox"]
                        face_gender = param["gender"]  # face gender
                        # predictions.append([dets[param["id"]]])  # prediction
                        local_predictions.append(dets[param["id"]])
                        face_embedding_list += [param["embed"]]
                        x_center = int((x1 + x2) / 2)  # set new center
                        y_center = int((y1 + y2) / 2)  # set new center
                        if not self.similarface:  # if user want to get only one face in frame
                            break
                if len(local_predictions) > 0:
                    predictions.append(local_predictions)
                else:
                    predictions.append([None])

        return predictions

    def face_detect_with_alignment_all(self, images):
        predictions = []
        for image in tqdm(images):
            dets = self.face_recognition.get_faces(image)
            if not dets:
                predictions.append([None])
                continue
            predictions.append(dets)
        return predictions

    def process_frame(self, args):
        frame, face_det_result, source_face, progress_bar = args
        tmp_frame = frame.copy()
        for face in face_det_result:
            if face is None:
                break
            else:
                tmp_frame = self.face_swap_model.get(tmp_frame, face, source_face, paste_back=True)

        with self.lock:
            self.progress += 1
            progress_bar.update(1)  # Update progress bar in a thread-safe manner

        return tmp_frame

    def swap_video(self, target_frames, source_face, target_face_fields, save_file: str, multiface=False, fps=30, video_format=".mp4"):
        if "CUDAExecutionProvider" in self.access_providers and torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu'):
            # thread will not work correct with GPU
            self.swap_video_cuda(target_frames, source_face, target_face_fields, save_file, multiface, fps, video_format)
        else:
            self.swap_video_thread(target_frames, source_face, target_face_fields, save_file, multiface, fps,video_format)

    def swap_video_thread(self, target_frames, source_face, target_face_fields, save_file: str,
                   multiface=False, fps=30, video_format=".mp4"):
        """
        Face swap video with Threads. Will not work with CUDA
        :param target_frames: target frames
        :param source_face: source face
        :param target_face_fields: crop field for target face
        :param save_file: save file path
        :param multiface: bool use swap all face or use target crop
        :param fps: video fps
        :param video_format: video format
        :return:
        """

        frame_h, frame_w = target_frames[0].shape[:-1]
        if video_format == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif video_format == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise ValueError("Unsupported video format: {}".format(video_format))

        out = cv2.VideoWriter(save_file, fourcc, fps, (frame_w, frame_h))

        if multiface:
            print("Getting all face...")
            face_det_results = self.face_detect_with_alignment_all(target_frames)
        else:
            print("Getting target face...")
            face_det_results = self.face_detect_with_alignment_crop(target_frames, target_face_fields)

        print("Starting face swap...")

        progress_bar = tqdm(total=len(target_frames), unit='it', unit_scale=True)

        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(self.process_frame, [
                (frame, face_det_results[i], source_face, progress_bar) for i, frame in enumerate(target_frames)
            ]))

        progress_bar.close()

        for frame in processed_frames:
            out.write(frame)

        out.release()
        print("Face swap processing finished...")

    def swap_video_cuda(self, target_frames, source_face, target_face_fields, save_file: str, multiface=False, fps=30, video_format=".mp4"):
        """Face swap video without Threads"""

        frame_h, frame_w = target_frames[0].shape[:-1]

        if video_format == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif video_format == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise ValueError("Unsupported video format: {}".format(video_format))

        out = cv2.VideoWriter(save_file, fourcc, fps, (frame_w, frame_h))

        if multiface:
            print("Getting all face...")
            face_det_results = self.face_detect_with_alignment_all(target_frames)
        else:
            print("Getting target face...")
            face_det_results = self.face_detect_with_alignment_crop(target_frames, target_face_fields)

        print("Starting face swap...")
        progress_bar = tqdm(total=len(face_det_results), unit='it', unit_scale=True)
        for i, dets in enumerate(face_det_results):
            tmp_frame = target_frames[i].copy()
            for face in dets:
                if face is None:
                    break
                else:
                    tmp_frame = self.face_swap_model.get(tmp_frame, face, source_face, paste_back=True)
            out.write(tmp_frame)
            progress_bar.update(1)
        else:
            out.release()
            print("Face swap processing finished...")
        progress_bar.close()


    def swap_image(self, target_frame, source_face, face_fields, save_dir: str, multiface=False):
        save_file = os.path.join(save_dir, "swapped_image.png")
        x_center, y_center = self.get_real_crop_box(target_frame, face_fields)
        dets = self.face_recognition.get_faces(target_frame)
        if not dets:
            raise FaceNotDetectedError("Face is not detected in target image!")

        if x_center is None or y_center is None or multiface:
            for face in dets:
                target_frame = self.face_swap_model.get(target_frame, face, source_face, paste_back=True)
            else:
                cv2.imwrite(save_file, target_frame)
                return target_frame
        else:
            for face in dets:
                x1, y1, x2, y2 = face.bbox
                if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                    target_frame = self.face_swap_model.get(target_frame, face, source_face, paste_back=True)
                    cv2.imwrite(save_file, target_frame)
                    return target_frame

        raise FaceNotDetectedError("Face is not detected in user crop field in target image!")


class FaceNotDetectedError(Exception):
    pass
