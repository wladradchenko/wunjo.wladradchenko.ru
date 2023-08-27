import face_alignment
import numpy as np
from tqdm import tqdm
import torch
import dlib
import cv2
import sys
import os


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "deepfake"))
from src.video2fake import Wav2Lip
from src.face3d.recognition import FaceRecognition
sys.path.pop(0)


def get_frames(video: str, rotate: int, crop: list, resize_factor: int):
    """
    Extract frames from a video, apply resizing, rotation, and cropping.

    :param video: path to the video file
    :param rotate: number of 90-degree rotations
    :param crop: list with cropping coordinates [y1, y2, x1, x2]
    :param resize_factor: factor by which the frame should be resized
    :return: list of processed frames, fps of the video
    """
    print("Start reading video")

    video_stream = cv2.VideoCapture(video)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []

    try:
        while True:
            still_reading, frame = video_stream.read()

            if not still_reading:
                break

            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            for _ in range(rotate):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            x2 = x2 if x2 != -1 else frame.shape[1]
            y2 = y2 if y2 != -1 else frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    finally:
        video_stream.release()

    print(f"Number of frames available for inference: {len(full_frames)}")

    return full_frames, fps


class GenerateFakeVideo2Lip:
    def __init__(self, model_path):
        self.face_recognition = FaceRecognition(model_path)
        self.face_fields = None

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

    def get_real_crop_box(self, image):
        """

        :param image:
        :return:
        """
        if self.face_fields is not None:
            # user crop face in frontend
            squareFace = self.face_fields[0]
            # real size
            originalHeight, originalWidth, _ = image.shape

            canvasWidth = squareFace["canvasWidth"]
            canvasHeight = squareFace["canvasHeight"]
            # Calculate the scale factor
            scaleFactor = min(originalWidth / canvasWidth, originalHeight / canvasHeight)

            # Calculate the new position and size of the square face on the original image
            if originalWidth > originalHeight:
                offsetX = (originalWidth - originalHeight) / 2
                newX1 = offsetX + (squareFace['x']) * scaleFactor
                newX2 = offsetX + (squareFace['x'] + squareFace['width']) * scaleFactor
                newY1 = squareFace['y'] * scaleFactor
                newY2 = (squareFace['y'] + squareFace['height']) * scaleFactor
            else:
                offsetY = (originalHeight - originalWidth) / 2
                newX1 = squareFace['x'] * scaleFactor
                newX2 = (squareFace['x'] + squareFace['width']) * scaleFactor
                newY1 = offsetY + (squareFace['y']) * scaleFactor
                newY2 = offsetY + (squareFace['y'] + squareFace['height']) * scaleFactor

            d_user_crop = dlib.rectangle(int(newX1), int(newY1), int(newX2), int(newY2))
            # get the center point of d_new
            user_crop_center = d_user_crop.center()

            return user_crop_center.x, user_crop_center.y
        return None, None

    def face_detect_with_alignment(self, images, device, pads, nosmooth):
        predictions = []
        face_embedding_list = []
        face_gender = None
        x_center, y_center = self.get_real_crop_box(images[0])

        for image in tqdm(images):
            dets = self.face_recognition.get_faces(image)
            if not dets:
                predictions.append(None)
                continue
            # this is init first face
            if x_center is None or y_center is None:
                face = dets[0]  # get first face
                x1, y1, x2, y2 = face.bbox
                face_gender = face.gender  # face gender
                predictions.append([x1, y1, x2, y2])  # prediction
                face_embedding_list += [face.normed_embedding]
                x_center = int((x1 + x2) / 2)  # set new center
                y_center = int((y1 + y2) / 2)  # set new center
            elif not face_embedding_list:  # not face yet, set new face
                for face in dets:
                    x1, y1, x2, y2 = face.bbox
                    if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                        face_gender = face.gender  # face gender
                        predictions.append([x1, y1, x2, y2])  # prediction
                        face_embedding_list += [face.normed_embedding]
                        x_center = int((x1 + x2) / 2)  # set new center
                        y_center = int((y1 + y2) / 2)  # set new center
                        break
            else:  # here is already recognition
                local_face_param = []
                for face in dets:  # TODO can not be art use hasattr
                    x1, y1, x2, y2 = face.bbox
                    x_center = int((x1 + x2) / 2)  # set new center
                    y_center = int((y1 + y2) / 2)  # set new center
                    normed_embedding = face.normed_embedding
                    is_similar = self.face_recognition.is_similar_face(normed_embedding, face_embedding_list)
                    if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                        local_face_param += [{
                            "is_center": True, "is_gender": face_gender == face.gender, "is_embed": is_similar,
                            "bbox": face.bbox, "gender": face.gender, "embed": normed_embedding
                        }]
                    else:
                        local_face_param += [{
                            "is_center": False, "is_gender": face_gender == face.gender, "is_embed": is_similar,
                            "bbox": face.bbox, "gender": face.gender, "embed": normed_embedding
                        }]

                for param in local_face_param:
                    if (param["is_center"] or param["is_gender"]) and param["is_embed"]:
                        # this predicted
                        x1, y1, x2, y2 = param["bbox"]
                        face_gender = param["gender"]  # face gender
                        predictions.append([x1, y1, x2, y2])  # prediction
                        face_embedding_list += [param["embed"]]
                        x_center = int((x1 + x2) / 2)  # set new center
                        y_center = int((y1 + y2) / 2)  # set new center
                        break
                else:
                    predictions.append(None)

        smooth_windows_size = 5
        results = []
        pady1, pady2, padx1, padx2 = pads
        for rect, image in zip(predictions, images):
            if rect is None:
                results.append([0, 0, 1, 1])
            else:
                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)
                results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=smooth_windows_size)

        results = [[image[int(y1): int(y2), int(x1):int(x2)], (int(y1), int(y2), int(x1), int(x2))] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        return results


    def datagen(self, frames: list, mels: list, box: list, static: bool, img_size: int, wav2lip_batch_size: int,
                device: str = 'cpu', pads: list =[0, 10, 0, 0], nosmooth: bool = False):
        """
        Generator function that processes frames and their corresponding mel spectrograms
        for feeding into a Wav2Lip model.
        :param frames: List of input face frames (images).
        :param mels: List of mel spectrogram chunks corresponding to each frame.
        :param box: A bounding box [y1, y2, x1, x2] or [-1] for automatic face detection.
        :param static: Whether to use only the first frame for all mels.
        :param img_size: The target size to which detected faces will be resized.
        :param wav2lip_batch_size: Batch size for the Wav2Lip model.
        :param device: Device for torch (e.g., 'cuda', 'cpu').
        :param pads: Padding for the face bounding box.
        :param nosmooth: Whether to apply a smoothing function to detected bounding boxes.
        :yield: Batches of processed images, mel spectrograms, original frames, and face coordinates.
        """

        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if box[0] == -1:
            if not static:
                face_det_results = self.face_detect_with_alignment(frames, device, pads, nosmooth)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect_with_alignment([frames[0]], device, pads, nosmooth)
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (img_size, img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _load(self, checkpoint_path, device):
        """Load mode by torch"""
        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint


    def load_model(self, path, device):
        """Load model Wav2Lip"""
        # load wav2lip
        wav2lip = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path, device)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        wav2lip.load_state_dict(new_s)
        model = wav2lip.to(device)
        return model.eval()


    def generate_video_from_chunks(self, gen, mel_chunks, batch_size, wav2lip_checkpoint, device, save_dir, fps=30,
                                   video_name_without_format='wav2lip_video',video_format='.mp4'):
        """
        Generate a video from audio and face frames using a trained Wav2Lip model.
        :param gen: Generator that produces (img_batch, mel_batch, frames, coords).
        :param mel_chunks: List of mel spectrogram chunks.
        :param batch_size: Batch size for processing.
        :param wav2lip_checkpoint: Path to the Wav2Lip model checkpoint.
        :param device: Device for torch (e.g., 'cuda', 'cpu').
        :param save_dir: Directory where the result video will be saved.
        :param fps: Frames per second for the resulting video.
        :param video_name_without_format: Video name.
        :param video_format: Format of the video, '.mp4' or '.avi'.
        :return: path to generated video or None.
        """
        video_path = None
        # in order to find strange coord for mouth
        coords_mouth = []
        len_coords_mouth = 3  # find mean center between num coords mouth
        max_distance_mouth = 20

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):

            # Load model only once and set record for first frame
            if i == 0:
                model = self.load_model(wav2lip_checkpoint, device)
                print("Model loaded")

                frame_h, frame_w = frames[0].shape[:-1]

                if video_format == '.mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif video_format == '.avi':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                else:
                    raise ValueError("Unsupported video format: {}".format(video_format))

                video_path = os.path.join(save_dir, video_name_without_format + video_format)
                out = cv2.VideoWriter(video_path, fourcc, fps, (frame_w, frame_h))

            # Prepare the batches for the model
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            # Predict lip sync
            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                center_current = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Calculate the center of the current bounding box
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                # Check if the current bounding box is non-empty
                if y1 != 0 and y2 != 1 and x1 != 0 and x2 != 1:
                    # If less than 5 coordinates are stored, add the new one
                    if len(coords_mouth) < len_coords_mouth:
                        coords_mouth.append(center_current)
                        f[y1:y2, x1:x2] = p
                    else:
                        # Calculate distances between the current center and the centers stored in keep_coords
                        distances = [self.face_recognition.calculate_distance(center_current, prev_center) for prev_center in coords_mouth]
                        # If the current center is near any of the previous centers, update the frame
                        # if any(distance < max_distance for distance in distances):
                        if np.mean(distances) < max_distance_mouth:
                            f[y1:y2, x1:x2] = p
                        coords_mouth.pop(0)  # Remove the oldest center
                        coords_mouth.append(center_current)  # Add the current center
                out.write(f)
        else:
            out.release()

        return video_path