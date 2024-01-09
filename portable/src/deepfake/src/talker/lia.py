import os
import sys
import cv2
import uuid
import torch
import numpy as np
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "deepfake"))
from src.talker.encoder import Encoder
from src.talker.styledecoder import Synthesis
from src.face3d.recognition import FaceRecognition
sys.path.pop(0)


class FaceNotDetectedError(Exception):
    pass


class Lia(torch.nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1., blur_kernel=[1, 3, 3, 1]):
        super(Lia, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        img_recon = self.dec(wa, alpha, feats)

        return img_recon


class Talker:
    def __init__(self, model_path, talker_model, device="cpu"):
        self.device = device
        self.face_recognition = FaceRecognition(model_path)
        self.lia = self.load_lia(talker_model, device)
        self.frame = self.get_gradient_frame()

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

    def process_source(self, source, face_fields=None, size=512, crop_scale=1.6):
        frame = cv2.imread(source)
        face = self.face_detect_with_alignment_from_source_frame(frame, face_fields=face_fields)
        aimg, mat = get_cropped_head(frame, face.kps, size=size, scale=crop_scale)
        aimg = np.transpose(aimg[:, :, ::-1], (2, 0, 1)) / 255
        aimg_tensor = torch.from_numpy(aimg).unsqueeze(0).float()
        aimg_tensor_norm = (aimg_tensor - 0.5) * 2.0
        return aimg_tensor_norm, frame, mat

    def driven(self, save_dir, driving_path, source, face_fields=None):
        aimg_tensor_norm, frame_src, mat = self.process_source(source, face_fields)
        aimg_tensor_norm = aimg_tensor_norm.to(self.device)
        cap = cv2.VideoCapture(driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        file_name = str(uuid.uuid4()) + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame_src.shape[:2]
        save_path = os.path.join(save_dir, file_name)
        out = cv2.VideoWriter(save_path, fourcc, int(fps), (w, h))

        with torch.no_grad():
            h_start = None
            for index in tqdm(range(total_frames), desc='[ Render face ]', unit='frame'):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (512, 512))
                    frame_np = np.expand_dims(frame, axis=0).astype('float32')
                    frame_np = (frame_np / 255 - 0.5) * 2.0
                    frame_np = frame_np.transpose(0, 3, 1, 2)
                    frame_torch = torch.from_numpy(frame_np).to(self.device)
                    if index == 0:
                        h_start = self.lia.enc.enc_motion(frame_torch)
                    frame_recon = self.lia(aimg_tensor_norm, frame_torch, h_start)
                    frame_recon_np = frame_recon.permute(0, 2, 3, 1).cpu().numpy()[0]
                    frame_recon_np = ((frame_recon_np[:, :, ::-1] + 1) / 2) * 255
                    pasted = self.paste_back(frame_src, frame_recon_np, mat)
                    out.write(pasted.astype('uint8'))
                else:
                    break

        cap.release()
        out.release()
        return save_path

    def paste_back(self, img, face, matrix):
        inverse_affine = cv2.invertAffineTransform(matrix)
        h, w = img.shape[0:2]
        face_h, face_w = face.shape[0:2]
        inv_restored = cv2.warpAffine(face, inverse_affine, (w, h))
        inv_restored = inv_restored.astype('float32')
        mask = self.frame.copy().astype('float32') / 255
        mask = cv2.resize(mask, (face_w, face_h))
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
        img = inv_mask * inv_restored + (1 - inv_mask) * img
        return img.clip(0, 255).astype('uint8')

    @staticmethod
    def get_gradient_frame():
        # Define the size of the image
        height, width = 256, 256
        # Create a blank white image
        frame = np.zeros((height, width, 3), dtype=np.uint8) * 255
        # Define the size of the center white region
        center_size = int(0.8 * min(height, width))
        # Calculate the border size
        border_size = (min(height, width) - center_size) // 2
        # Create a black rectangle in the center
        frame[border_size:border_size + center_size, border_size:border_size + center_size] = [255, 255, 255]
        # Apply Gaussian blur to create a smooth gradient from white to black
        blurred_frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=10)
        return blurred_frame

    @staticmethod
    def get_real_crop_box(frame, squareFace):
        """
        Get real crop box
        :param frame:
        :return:
        """
        if squareFace is not None:
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
            newX2 = (squareFace['x'] + 1) * scaleFactorX  # 1 is width
            newY1 = squareFace['y'] * scaleFactorY
            newY2 = (squareFace['y'] + 1) * scaleFactorY  # 1 is height

            # Calculate center point
            center_x = (newX1 + newX2) / 2
            center_y = (newY1 + newY2) / 2

            return int(center_x), int(center_y)
        return None, None

    @staticmethod
    def load_lia(model_path, device="cpu"):
        gen = Lia(512, 512, 20, 1).to(device)
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        gen.load_state_dict(weight)
        gen.eval()
        return gen

def align_crop(img, landmark, size):
    template_ffhq = np.array([[192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]])
    template_ffhq *= (512 / size)
    matrix = cv2.estimateAffinePartial2D(landmark, template_ffhq, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix


def get_cropped_head(img, landmark, scale=1.4, size=512):
    center = np.mean(landmark, axis=0)
    landmark = center + (landmark - center) * scale
    return align_crop(img, landmark, size)