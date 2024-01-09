import numpy as np
from tqdm import tqdm
import torch
import cv2
import sys
import os


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "deepfake"))
from src.video2fake import Wav2Lip, Emo2Lip
from src.face3d.recognition import FaceRecognition
sys.path.pop(0)


class GenerateWave2Lip:
    def __init__(self, model_path, emotion_label, similar_coeff=0.95):
        self.face_recognition = FaceRecognition(model_path)
        self.face_fields = None
        self.emotion, self.use_emotion = (self.to_emotion_categorical(emotion_label), True) if emotion_label else (None, False)
        self.similar_coeff = 0.95  # Similarity coefficient for face

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
            squareFace = self.face_fields
            # real size
            originalHeight, originalWidth, _ = image.shape

            canvasWidth = squareFace["canvasWidth"]
            canvasHeight = squareFace["canvasHeight"]
            # Calculate the scale factor
            scaleFactorX = originalWidth / canvasWidth
            scaleFactorY = originalHeight / canvasHeight

            # Calculate the new position and size of the square face on the original image
            # Convert canvas square face coordinates to original image coordinates
            newX1 = squareFace['x'] * scaleFactorX
            newX2 = (squareFace['x'] + 1) * scaleFactorX
            newY1 = squareFace['y'] * scaleFactorY
            newY2 = (squareFace['y'] + 1) * scaleFactorY

            # Calculate center point
            center_x = (newX1 + newX2) / 2
            center_y = (newY1 + newY2) / 2

            return int(center_x), int(center_y)
        return None, None

    def face_detect_with_alignment_crop(self, image_files):
        """
        Detect faces to swap if face_fields
        :param image_files: list of file paths of target images
        :return:
        """
        predictions = []
        face_embedding_list = []
        face_gender = None

        # Read the first image to get the center
        first_image = cv2.imread(image_files[0])
        x_center, y_center = self.get_real_crop_box(first_image)

        for image_file in tqdm(image_files):
            image = cv2.imread(image_file)
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
                    is_similar = self.face_recognition.is_similar_face(normed_embedding, face_embedding_list, self.similar_coeff)
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
                        local_predictions.append(dets[param["id"]])
                        face_embedding_list += [param["embed"]]
                        x_center = int((x1 + x2) / 2)  # set new center
                        y_center = int((y1 + y2) / 2)  # set new center
                        break
                if len(local_predictions) > 0:
                    predictions.append(local_predictions)
                else:
                    predictions.append([None])

        return predictions


    def datagen(self, frame_files: list, mels: list, img_size: int, wav2lip_batch_size: int, pads: list =[0, 10, 0, 0]):
        """
        Generator function that processes frames and their corresponding mel spectrograms
        for feeding into a Wav2Lip model.
        :param frame_files: List of input path to images.
        :param mels: List of mel spectrogram chunks corresponding to each frame.
        :param img_size: The target size to which detected faces will be resized.
        :param wav2lip_batch_size: Batch size for the Wav2Lip model.
        :param pads: Padding for the face bounding box.
        :param nosmooth: Whether to apply a smoothing function to detected bounding boxes.
        :yield: Batches of processed images, mel spectrograms, original frames, and face coordinates.
        """

        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        face_det_results = self.face_detect_with_alignment_crop(frame_files)  # BGR2RGB for CNN face detection

        for i, m in enumerate(mels):
            idx = i % len(frame_files)

            frame_to_save = cv2.imread(frame_files[idx])
            dets = face_det_results[idx]
            coords = dets[0].get("bbox") if dets[0] is not None else (0, 0, 0, 0)
            x1, y1, x2, y2 = map(int, coords)
            x1 = max(x1, 0)
            x2 = max(x2, 0)
            y1 = max(y1, 0)
            y2 = max(y2, 0)
            if int(x2 - x1) == 0 and int(y2 - y1) == 0:
                # if empty face add as face full frame
                img_batch.append(cv2.resize(frame_to_save, (img_size, img_size)))
            else:
                face = frame_to_save[int(y1): int(y2), int(x1):int(x2)]
                face = cv2.resize(face, (img_size, img_size))
                img_batch.append(face)

            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append((y1, y2, x1, x2))

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
        wav2lip = Emo2Lip() if self.use_emotion else Wav2Lip()
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
                if self.use_emotion:
                    emotion = self.emotion.unsqueeze(0).to(device).repeat(img_batch.shape[0], 1)
                    pred = model(mel_batch, img_batch, emotion)
                else:
                    pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                if int(x2 - x1) == 0 and int(y2 - y1) == 0:
                    # save clear frame
                    out.write(f)
                else:
                    # save processed frame
                    center_current = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Calculate the center of the current bounding box
                    p = cv2.resize(p.astype(np.uint8), (int(x2 - x1), int(y2 - y1)))
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

    @staticmethod
    def to_emotion_categorical(y, num_classes=6, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y)
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return torch.tensor(categorical)
