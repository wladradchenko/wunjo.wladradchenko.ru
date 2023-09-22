import os
import cv2
import time
import face_alignment
import numpy as np
from PIL import Image
from tqdm import tqdm


class KeypointExtractor():
    def __init__(self, device):
        # face_alignment.LandmarksType.TWO_D is 1, before was face_alignment.LandmarksType.2_D
        # i think maybe the best way will be set 1, because developer of lib renames parameters
        self.detector = face_alignment.FaceAlignment(1, device=device)   

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
                    keypoints = self.detector.get_landmarks_from_image(np.array(images))[0]
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
