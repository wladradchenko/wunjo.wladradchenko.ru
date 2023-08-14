from facenet_pytorch import MTCNN, InceptionResnetV1
import face_alignment
import numpy as np
from tqdm import tqdm
import torch
import dlib
import cv2
import sys
import os


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


class FaceRecognition:
    def __init__(self):
        # Create an InceptionResnetV1 feature extractor object
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def detect_and_embed(self, image):
        # Convert the numpy image to a PyTorch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Add the batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Normalize as per ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Calculate embedding
        embedding = self.resnet(image_tensor)
        return embedding

    def is_similar_face(self, embedding, face_pred_list):
        """
        Check if the current embedding is similar to any in the face_pred_list.
        :param  embedding: (torch.Tensor): The embedding of the current face.
        :param face_pred_list: List of keep faces
        :return: bool True if similar face is found, False otherwise.
        """
        threshold = 0.8  # You might need to adjust this based on your requirements

        for known_embedding in face_pred_list:
            distance = torch.dist(embedding, known_embedding, p=2)  # Compute L2 distance
            if distance < threshold:
                return True

        return False


class GenerateFakeVideo2Lip:
    def __init__(self):
        self.face_recognition = FaceRecognition()
        self.face_fields = None
        self.face_detected_in_fields = False
        self.face_prediction_list = []

    def get_smoothened_boxes(self, boxes: list, T: int = 5):
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

    def face_detect_with_alignment(self, images, device, pads, nosmooth):
        smooth_windows_size = 5
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

        predictions = []
        for image in tqdm(images):
            dets = fa.get_landmarks(image)
            if dets is None or len(dets) == 0:
                predictions.append(None)
            else:
                # Using the detected landmarks, get the bounding box
                if self.face_fields is None:
                    print("Not set crop, take first face")
                    landmarks = dets[0]  # get first face
                    x1, y1 = np.min(landmarks, axis=0).astype(int)
                    x2, y2 = np.max(landmarks, axis=0).astype(int)
                    predictions.append([x1, y1, x2, y2])
                    self.face_fields = [{"x_center": int((x1+x2)/2), "y_center": int((y1+y2)/2)}]
                    # recognition face
                    cropped_face = image[y1:y2, x1:x2]
                    if y1 < 0 or y2 < 0 or x1 < 0 or x2 < 0:
                        continue
                    cropped_face_resized = cv2.resize(cropped_face, (160, 160))
                    encoding = self.face_recognition.detect_and_embed(cropped_face_resized)
                    if encoding is not None:
                        self.face_prediction_list += encoding  # TODO check format
                else:
                    # user crop face in frontend
                    squareFace = self.face_fields[0]

                    if squareFace.get("x_center") is None and squareFace.get("y_center") is None:
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
                        print("Change face field param")
                        self.face_fields = [{"x_center": user_crop_center.x, "y_center": user_crop_center.y}]

                    # check if user_crop_center is inside det
                    for landmarks in dets:
                        x1, y1 = np.min(landmarks, axis=0).astype(int)
                        x2, y2 = np.max(landmarks, axis=0).astype(int)
                        # Check if user_crop_center is inside the bounding box defined by (x1, y1) and (x2, y2)
                        if x1 <= self.face_fields[0]["x_center"] <= x2 and y1 <= self.face_fields[0]["y_center"] <= y2:
                            print("Center is inside bounding box derived from landmarks.")
                            predictions.append([x1, y1, x2, y2])
                            self.face_fields = [{"x_center": int((x1+x2)/2), "y_center": int((y1+y2)/2)}]
                            # recognition face
                            cropped_face = image[y1:y2, x1:x2]
                            if y1 < 0 or y2 < 0 or x1 < 0 or x2 < 0:
                                continue
                            cropped_face_resized = cv2.resize(cropped_face, (160, 160))
                            encoding = self.face_recognition.detect_and_embed(cropped_face_resized)
                            if encoding is not None:
                                self.face_prediction_list += encoding  # TODO check format
                            break
                    else:
                        if self.face_detected_in_fields:
                            # TODO try to use face recognition to find similiar face
                            for landmarks in dets:
                                x1, y1 = np.min(landmarks, axis=0).astype(int)
                                x2, y2 = np.max(landmarks, axis=0).astype(int)
                                # recognition face
                                cropped_face = image[y1:y2, x1:x2]
                                if y1 < 0 or y2 < 0 or x1 < 0 or x2 < 0:
                                    continue
                                cropped_face_resized = cv2.resize(cropped_face, (160, 160))
                                encoding = self.face_recognition.detect_and_embed(cropped_face_resized)
                                if encoding is not None:
                                    is_similar = self.face_recognition.is_similar_face(encoding, self.face_prediction_list)
                                    if is_similar:
                                        print("Recognition new similar face")
                                        predictions.append([x1, y1, x2, y2])
                                        self.face_fields = [{"x_center": int((x1 + x2) / 2), "y_center": int((y1 + y2) / 2)}]
                                        self.face_prediction_list += encoding  # TODO check format
                                        break
                            else:
                                # not similar faces
                                predictions.append(None)
                        else:
                            # not detected yet on first frames
                            predictions.append(None)

        results = []
        pady1, pady2, padx1, padx2 = pads
        for rect, image in zip(predictions, images):
            if rect is None:
                print('Face not detected! Ensure the video contains a face in all frames.')
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

        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
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
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, f"{root_path}/deepfake")
        from src.deepfake.video2fake import Wav2Lip
        sys.path.pop(0)

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
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)
        else:
            out.release()

        return video_path