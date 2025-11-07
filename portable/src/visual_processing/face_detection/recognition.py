# coding: utf-8

"""
face detectoin and alignment using InsightFace
"""
import pickle
import os.path
import subprocess

import numpy as np
import soundfile as sf
from insightface.app import FaceAnalysis
from insightface.app.common import Face
import time

import torch
import onnxruntime

import os.path as osp
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from tqdm import tqdm
from .sync_net import SyncNetInstance
from scipy import signal
from scenedetect import open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        return self.diff

    def clear(self):
        self.start_time = 0.
        self.diff = 0.


def sort_by_direction(faces, direction: str = 'large-small', face_center=None, gender=None, threshold="inf"):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]), reverse=True)
    if direction == 'distance-from-retarget-face':  # euclidean_distance
        return sorted(faces, key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    if direction == 'distance-from-embedding':
        # First, filter the faces based on gender and dynamic threshold
        # Then sort the filtered faces by their mean distance
        # TODO At the same time bug with gender detect: return sorted(filter(lambda face: face.gender == gender and face.mean_distance < threshold * face.dynamic_threshold, faces), key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
        return sorted(filter(lambda face: face.mean_distance < threshold * face.dynamic_threshold, faces), key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    return faces


class FaceRecognition(FaceAnalysis):
    def __init__(self, name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)

        self.timer = Timer()
        self.id = 0
        self.window = {}  # List to store previous face detections
        self.window_size = 50  # last n distances for sliding window
        self.max_window_size = 250  # total embedding list in RAM
        self.running_average = {}
        self.max_rate_of_change = 0.05  # maximum allowed rate of change for running average

    def set_face_id(self):
        if self.window.get(self.id) is None:
            self.window[self.id] = []
            self.running_average[self.id] = None

    # Method to update memory of detected faces
    def update_memory(self, face):
        # Use id of face
        self.set_face_id()

        # Store the center of the face and any other relevant info
        center = ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)
        self.window[self.id].append({'embedding': face.normed_embedding, 'center': center, 'gender': face.gender})

        # Optionally, limit the size of the memory
        if len(self.window[self.id]) > self.max_window_size:  # Keep only the last 10 detected faces
            self.window[self.id].pop(0)

    @staticmethod
    def get_dynamic_threshold(face, height):
        # Define thresholds for different face sizes
        face_height = (face.bbox[3] - face.bbox[1]) / height
        if face_height < 0.1:
            return 1.05
        if face_height < 0.2:
            return 1.2
        if face_height < 0.4:
            return 1.3
        elif face_height < 0.8:
            return 1.35
        elif face_height < 0.9:
            return 1.4
        else:
            return 1.45

    @staticmethod
    def get_center(crop: dict = None):
        """
        Get center coord
        :return:
        """
        if crop is not None:
            return int(crop["x"] + crop["width"] / 2), int(crop["y"] + crop["height"] / 2)
        return None, None

    def clear(self):
        self.window = {}
        self.running_average = {}

    def update_center(self, crop: dict = None):
        if crop is None:
            return
        # Use id of face
        self.set_face_id()
        # Update window
        if len(self.window[self.id]) > 0:
            print("Update center")
            face_center = self.get_center(crop)
            self.window[self.id][-1]["center"] = face_center

    def get_mouth(self, img_bgr):
        # Find opening mouth
        faces = self.get(img_bgr)
        for face in faces:
            landmarks = face.landmark_2d_106  # 106 key points
            upper_lip = landmarks[52]  # Example: a dot on the upper lip
            lower_lip = landmarks[58]  # Example: a dot on the lower lip
            mouth_openness = abs(upper_lip[1] - lower_lip[1]) / (face.bbox[3] - face.bbox[1])  # Normalization
            return mouth_openness
        return 0

    def get(self, img_bgr, **kwargs):
        # Get params
        max_num = kwargs.get('max_face_num', 0)  # the number of the detected faces, 0 means no limit
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # whether to do 106-point detection
        direction = kwargs.get('direction', 'large-small')  # sorting direction
        id_face = kwargs.get('id', None)
        if id_face is not None and isinstance(id_face, int):
            self.id = id_face
        crop = kwargs.get('crop', None)
        # Use id of face
        self.set_face_id()
        # Get face params
        face_center = self.get_center(crop) if not len(self.window[self.id]) > 0 else self.window[self.id][-1]["center"]
        face_gender = np.argmax(np.bincount([known_face["gender"] for known_face in self.window[self.id]])) if len(self.window[self.id]) > 0 else None
        height, _, _ = img_bgr.shape

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue

                if (not flag_do_landmark_2d_106) and taskname == 'landmark_2d_106':
                    continue

                # print(f'taskname: {taskname}')
                model.get(img_bgr, face)
            # Find the most similar face based on embeddings
            face.mean_distance = np.mean([self.calculate_distance(face.normed_embedding, known_face["embedding"]) for known_face in self.window[self.id]]) if len(self.window[self.id]) > 0 else float('inf')
            face.dynamic_threshold = self.get_dynamic_threshold(face, height=height)
            # if self.running_average[self.id] and face.mean_distance:  # TODO debug for difference face sizes
            #     print(i, face.dynamic_threshold, face.mean_distance / self.running_average[self.id])
            ret.append(face)

        ret = sort_by_direction(
            ret,
            'distance-from-retarget-face' if direction == 'distance-from-embedding' and len(self.window[self.id]) < self.window_size else direction,
            face_center,
            face_gender,
            self.running_average[self.id] if self.running_average[self.id] else "inf"
        )
        if len(self.window[self.id]) == 0 and face_center[0] and face_center[1]:
            for face in ret:
                x1, y1, x2, y2 = face.bbox
                if x1 <= face_center[0] <= x2 and y1 <= face_center[1] <= y2:
                    ret = [face]
                    break
            else:
                ret = []

        if len(ret) > 0 and direction == 'distance-from-embedding':
            self.update_memory(ret[0])
            if len(self.window[self.id]) > self.window_size - 1:
                if self.running_average[self.id] is None:
                    self.running_average[self.id] = max([self.calculate_distance(ret[0].normed_embedding, known_face["embedding"]) for known_face in self.window[self.id]])
                else:
                    # Limit the rate of change for running_average
                    change = max([self.calculate_distance(ret[0].normed_embedding, known_face["embedding"]) for known_face in self.window[self.id]]) - self.running_average[self.id]
                    change = np.clip(change, -self.max_rate_of_change, self.max_rate_of_change)
                    self.running_average[self.id] += change
            if len(self.window[self.id]) > self.max_window_size:
                # Remove first element to be equal self.window_size - 1
                self.window[self.id] = self.window[self.id][1:]

        return ret

    def warmup(self):
        self.timer.tic()

        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

        elapse = self.timer.toc()
        print(f'FaceAnalysis warmup time: {elapse:.3f}s')

    @staticmethod
    def calculate_distance(embedding1, embedding2):
        # Your distance calculation here
        return np.linalg.norm(embedding1 - embedding2)


class FaceRecognitionProcessing:
    @staticmethod
    def load(model_path, device="cpu"):
        # use cpu as with cuda on onnx can be problem
        access_providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in access_providers:
            provider = ["CUDAExecutionProvider"] if torch.cuda.is_available() and device != "cpu" else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        face_analysis_wrapper = FaceRecognition(
            name="buffalo_l",
            root=model_path,
            providers=provider,
        )

        face_analysis_wrapper.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.1)
        face_analysis_wrapper.warmup()
        return face_analysis_wrapper

    @staticmethod
    def detect(face_analysis_wrapper, img_bgr):
        bboxes, kpss = face_analysis_wrapper.det_model.detect(img_bgr, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            return []
        return [[int(b) for b in box[0:4]] for box in bboxes.tolist()]

    @staticmethod
    def load_image_rgb(image_path: str):
        if not osp.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FaceRecognitionSpeaker:
    def __init__(self, progress_callback=None, **kwargs):
        self.device = kwargs.get("device")
        self.syncnet_v2 = kwargs.get("syncnet_v2")
        self.model_path = kwargs.get("model_path")
        self.progress_callback = progress_callback

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        """IOU FUNCTION"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def __call__(self, video_file: str, save_path: str, min_track: int = 100, threshold: float = 3.0, **kwargs):
        # Create work dirs
        work_dir = os.path.join(save_path, "pywork")
        os.makedirs(work_dir, exist_ok=True)

        file_dir = os.path.join(save_path, "pyfile")
        os.makedirs(file_dir, exist_ok=True)

        tmp_dir = os.path.join(save_path, "pytmp")
        os.makedirs(tmp_dir, exist_ok=True)

        crop_dir = os.path.join(save_path, "pycrop")
        os.makedirs(crop_dir, exist_ok=True)

        frames_dir = os.path.join(save_path, "pyframes")
        os.makedirs(frames_dir, exist_ok=True)

        # Get work files
        video_file = self.normalize_video(video_file, file_dir)
        self.extract_frames(video_file, frames_dir)
        audio_file = self.extract_audio(video_file, file_dir)

        # Get scenes
        scenes = self.get_scenes(video_file, work_dir)
        scenes_dict = {scene[0].get_frames(): {} for scene in scenes}
        faces = self.get_faces(video_file, work_dir)

        idx = 0

        print("Analysing scenes...")
        progress_bar = tqdm(total=len(scenes), unit='it', unit_scale=True)

        for scene in scenes:
            if scene[1].frame_num - scene[0].frame_num >= min_track:
                for track in self.set_tracks(faces[scene[0].frame_num:scene[1].frame_num], fidx=scene[0].frame_num):
                    crop = self.crop(audio_file, frames_dir, file=os.path.join(crop_dir, '%05d' % idx), track=track)
                    with open(os.path.join(work_dir, f'track_{idx}.pckl'), 'wb') as f:
                        pickle.dump(crop, f)

                    idx += 1
                    progress_bar.update(1)

        print("Finished analysing scenes...")
        progress_bar.close()

        syncnet = SyncNetInstance()
        syncnet.loadParameters(self.syncnet_v2)

        batch_dets = []
        flist = [os.path.join(crop_dir, f) for f in os.listdir(crop_dir) if f.startswith('0') and f.lower().endswith('.avi')]
        flist.sort()

        print("Analysing speach tracks and faces...")
        progress_bar = tqdm(total=len(scenes), unit='it', unit_scale=True)

        for idx, fname in enumerate(tqdm(flist)):
            offset, conf, dist = syncnet.evaluate(tmp_dir, video_file=fname)
            batch_dets.append(dist)
            progress_bar.update(1)

            mean_dists = np.mean(np.stack(dist, 1), 1)
            minidx = np.argmin(mean_dists, 0)
            minval = mean_dists[minidx]

            fdist = np.stack([d[minidx] for d in dist])
            fdist = np.pad(fdist, (3, 3), 'constant', constant_values=10)

            fconf = np.median(mean_dists) - fdist
            fconfm = signal.medfilt(fconf, kernel_size=9)

            with open(os.path.join(work_dir, f'track_{idx}.pckl'), 'rb') as f:
                track = pickle.load(f)

            frames = track['track']['frame'].tolist()
            for fidx, frame in enumerate(frames):
                offset_idx = fidx + track['track']['offset_idx']
                conf = fconfm[fidx]  # coef confidence from -10 to 10
                x = track['proc_track']['x'][fidx]
                y = track['proc_track']['y'][fidx]
                scene_dict = scenes_dict.get(offset_idx)

                if conf > threshold:  # and tidx != idx:
                    if scene_dict is None or scene_dict.get("conf") is None or scene_dict.get("conf") < conf:
                        scenes_dict[offset_idx] = {"x": x, "y": y, "idx": idx, "conf": conf}

            if self.progress_callback:
                self.progress_callback(round(idx / len(flist) * 100, 0), f"Combine tracks...")

        print("Finished analysing speakers...")
        progress_bar.close()

        filtered_scenes_dict = {}
        last_idx = object()
        [filtered_scenes_dict.update({k: v}) or (last_idx := v.get('idx') if v else None) for k, v in sorted(scenes_dict.items()) if (v.get('idx') if v else None) != last_idx]
        del syncnet

        return filtered_scenes_dict

    @staticmethod
    def normalize_video(video_path: str, save_path: str):
        save_file = os.path.join(save_path, 'video.avi')
        cmd = "ffmpeg -y -i %s -c:v libx264 -preset fast -crf 18 -r 25 -pix_fmt yuv420p %s" % (video_path, save_file)
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return save_file

    @staticmethod
    def extract_frames(video_path: str, save_path: str):
        cmd = "ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (video_path, os.path.join(save_path, '%06d.jpg'))
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def extract_audio(video_path: str, save_path: str):
        audio_file = os.path.join(save_path, 'audio.wav')
        cmd = "ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (video_path, audio_file)
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # If video was without sound
        if not os.path.exists(audio_file):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps if fps > 0 else 1.0
            cap.release()

            sr = 16000
            samples = int(duration * sr)
            silent_audio = np.zeros(samples, dtype=np.float32)  # float32 → WAV PCM 16
            sf.write(audio_file, silent_audio, sr, subtype='PCM_16')
        return audio_file

    def get_faces(self, video_path: str, work_dir: str, facedet_scale: float = 0.3) -> list:
        # use cpu as with cuda on onnx can be problem
        access_providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in access_providers:
            provider = ["CUDAExecutionProvider"] if torch.cuda.is_available() and self.device != "cpu" else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        face_analysis = FaceAnalysis(
            name="buffalo_l",
            root=self.model_path,
            providers=provider,
        )
        face_analysis.prepare(ctx_id=0, det_size=(512, 512), det_thresh=facedet_scale)

        cap = cv2.VideoCapture(video_path)
        dets = []
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print("Starting get faces...")
        progress_bar = tqdm(total=total_frames, unit='it', unit_scale=True)

        for fidx in range(int(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_analysis.get(image_np)
            frame_dets = [{'frame': fidx, 'bbox': face.bbox[:4], 'conf': float(face.det_score)} for face in faces]
            dets.append(frame_dets)
            progress_bar.update(1)
            if self.progress_callback:
                self.progress_callback(round(fidx / total_frames * 100, 0), f"Get faces...")
        cap.release()

        print("Get faces processing finished...")
        progress_bar.close()

        with open(os.path.join(work_dir, "face.pckl"), 'wb') as fil:
            pickle.dump(dets, fil)

        del face_analysis
        return dets

    @staticmethod
    def get_scenes(video_file: str, work_dir: str) -> list:
        video = open_video(video_file)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        # Add ContentDetector algorithm (constructor takes detector options like threshold).
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(frame_source=video)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            duration = video.duration
            scene_list = [(video.base_timecode, video.base_timecode + duration)]

        with open(os.path.join(work_dir, 'scene.pckl'), 'wb') as fil:
            pickle.dump(scene_list, fil)

        print(f'Scenes detected {len(scene_list)}')
        return scene_list

    def set_tracks(self, scene_faces: list, fidx: int, min_face_size: int = 100, min_track: int = 100, num_failed_det: int = 25,
                   iou_thres: float = 0.5):
        """
        min_face_size: Minimum face size in pixels
        min_track: Minimum facetrack duration
        num_failed_det: Number of missed detections allowed before tracking is stopped
        iou_thres: Minimum IOU between consecutive face detections
        """
        while any(scene_faces):
            track = []
            for faces in scene_faces:
                keep_idx = []
                for idx, face in enumerate(faces):
                    if not track:
                        track.append(face)
                    elif face['frame'] - track[-1]['frame'] <= num_failed_det:
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iou_thres:
                            track.append(face)
                            continue
                        else:
                            keep_idx.append(idx)
                    else:
                        keep_idx.append(idx)
                        break
                faces[:] = [faces[i] for i in keep_idx]

            if not track:
                break
            elif len(track) > min_track:
                framenum = np.array([f['frame'] for f in track])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                frame_i = np.arange(framenum[0], framenum[-1] + 1, dtype=np.int32)
                bboxes_i = np.stack([np.interp(frame_i, framenum, bboxes[:, ij]) for ij in range(4)], axis=1).astype(np.float32)

                if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > min_face_size:
                    yield {'frame': frame_i, 'bbox': bboxes_i, 'offset_idx': fidx}

    @staticmethod
    def crop(audio_path: str, frames_dir: str, file: str, track: dict, frame_rate: int = 25, crop_scale: float = 0.4):
        flist = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')]
        flist.sort()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vOut = cv2.VideoWriter(file + 't.avi', fourcc, frame_rate, (224, 224))

        dets = {'x': [], 'y': [], 's': []}
        for det in track['bbox']:
            dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)  # face crop radius
            dets['y'].append((det[1] + det[3]) / 2)  # crop center x
            dets['x'].append((det[0] + det[2]) / 2)  # crop center y

            # Smooth detections
        dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
        # TODO? Заменить ли на from scipy.ndimage import median_filter dets['s'] = median_filter(dets['s'], size=13, mode='nearest')
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

        for fidx, frame in enumerate(track['frame']):
            cs = crop_scale
            bs = dets['s'][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

            image = cv2.imread(flist[frame])
            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my = dets['y'][fidx] + bsi  # BBox center Y
            mx = dets['x'][fidx] + bsi  # BBox center X

            face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            vOut.write(cv2.resize(face, (224, 224)))

        audio_tmp = file + '.wav'
        audio_start = (track['frame'][0]) / frame_rate
        audio_end = (track['frame'][-1] + 1) / frame_rate

        vOut.release()

        # CROP AUDIO FILE
        cmd = "ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (audio_path, audio_start, audio_end, audio_tmp)
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # COMBINE AUDIO AND VIDEO FILES
        cmd = "ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (file, audio_tmp, file)
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.remove(audio_tmp)
        os.remove(file + 't.avi')
        # print('Mean pos: x %.2f y %.2f s %.2f' % (np.mean(dets['x']), np.mean(dets['y']), np.mean(dets['s'])))
        return {'track': track, 'proc_track': dets}
