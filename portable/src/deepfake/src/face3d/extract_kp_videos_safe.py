import os
import cv2
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model

from backend.folders import ALL_MODEL_FOLDER


class KeypointExtractor():
    def __init__(self, device='cuda'):

        ### gfpgan/weights
        root_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan/weights')

        self.detector = init_alignment_model('awing_fan', device=device, model_rootpath=root_path)
        self.det_net = init_detection_model('retinaface_resnet50', half=False,device=device, model_rootpath=root_path)

    def extract_keypoint(self, images, name=None, info=True):
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images,desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)

                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    with torch.no_grad():
                        # face detection -> face alignment.
                        img = np.array(images)
                        bboxes = self.det_net.detect_faces(images, 0.97)
                        
                        bboxes = bboxes[0]

                        img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]

                        keypoints = landmark_98_to_68(self.detector.get_landmarks(img)) # [0]

                        #### keypoints to the original location
                        keypoints[:,0] += int(bboxes[0])
                        keypoints[:,1] += int(bboxes[1])

                        break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break    
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)                    
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(
        images, 
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )
