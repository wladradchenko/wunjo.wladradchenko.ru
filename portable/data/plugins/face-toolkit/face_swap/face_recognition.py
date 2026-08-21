# coding: utf-8
"""Slim InsightFace wrapper used by LivePortrait cropping (no SyncNet/scene deps)."""

import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Timer(object):
    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        return self.diff

    def clear(self):
        self.start_time = 0.0
        self.diff = 0.0


def sort_by_direction(faces, direction: str = "large-small", face_center=None, gender=None, threshold=float("inf")):
    if len(faces) <= 0:
        return faces
    if direction == "left-right":
        return sorted(faces, key=lambda face: face["bbox"][0])
    if direction == "right-left":
        return sorted(faces, key=lambda face: face["bbox"][0], reverse=True)
    if direction == "top-bottom":
        return sorted(faces, key=lambda face: face["bbox"][1])
    if direction == "bottom-top":
        return sorted(faces, key=lambda face: face["bbox"][1], reverse=True)
    if direction == "small-large":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))
    if direction == "large-small":
        return sorted(
            faces,
            key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]),
            reverse=True,
        )
    if direction == "distance-from-retarget-face":
        return sorted(
            faces,
            key=lambda face: (
                ((face["bbox"][2] + face["bbox"][0]) / 2 - face_center[0]) ** 2
                + ((face["bbox"][3] + face["bbox"][1]) / 2 - face_center[1]) ** 2
            )
            ** 0.5,
        )
    if direction == "distance-from-embedding":
        return sorted(
            filter(lambda face: face.mean_distance < threshold * face.dynamic_threshold, faces),
            key=lambda face: (
                ((face["bbox"][2] + face["bbox"][0]) / 2 - face_center[0]) ** 2
                + ((face["bbox"][3] + face["bbox"][1]) / 2 - face_center[1]) ** 2
            )
            ** 0.5,
        )
    return faces


class FaceRecognition(FaceAnalysis):
    def __init__(self, name="buffalo_l", root="~/.insightface", allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)
        self.timer = Timer()
        self.id = 0
        self.window = {}
        self.window_size = 50
        self.max_window_size = 250
        self.running_average = {}
        self.max_rate_of_change = 0.05

    def set_face_id(self):
        if self.window.get(self.id) is None:
            self.window[self.id] = []
            self.running_average[self.id] = None

    def update_memory(self, face):
        self.set_face_id()
        center = ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)
        self.window[self.id].append({"embedding": face.normed_embedding, "center": center, "gender": face.gender})
        if len(self.window[self.id]) > self.max_window_size:
            self.window[self.id].pop(0)

    @staticmethod
    def get_dynamic_threshold(face, height):
        face_height = (face.bbox[3] - face.bbox[1]) / height
        if face_height < 0.1:
            return 1.05
        if face_height < 0.2:
            return 1.2
        if face_height < 0.4:
            return 1.3
        if face_height < 0.8:
            return 1.35
        if face_height < 0.9:
            return 1.4
        return 1.45

    @staticmethod
    def get_center(crop: dict = None):
        if crop is not None:
            return int(crop["x"] + crop["width"] / 2), int(crop["y"] + crop["height"] / 2)
        return None, None

    def clear(self):
        self.window = {}
        self.running_average = {}

    def update_center(self, crop: dict = None):
        if crop is None:
            return
        self.set_face_id()
        if len(self.window[self.id]) > 0:
            face_center = self.get_center(crop)
            self.window[self.id][-1]["center"] = face_center

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get("max_face_num", 0)
        flag_do_landmark_2d_106 = kwargs.get("flag_do_landmark_2d_106", True)
        direction = kwargs.get("direction", "large-small")
        id_face = kwargs.get("id", None)
        if id_face is not None and isinstance(id_face, int):
            self.id = id_face
        crop = kwargs.get("crop", None)
        self.set_face_id()
        face_center = self.get_center(crop) if not len(self.window[self.id]) > 0 else self.window[self.id][-1]["center"]
        face_gender = (
            np.argmax(np.bincount([known_face["gender"] for known_face in self.window[self.id]]))
            if len(self.window[self.id]) > 0
            else None
        )
        height, _, _ = img_bgr.shape

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric="default")
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None if kpss is None else kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == "detection":
                    continue
                if (not flag_do_landmark_2d_106) and taskname == "landmark_2d_106":
                    continue
                model.get(img_bgr, face)
            face.mean_distance = (
                np.mean(
                    [
                        self.calculate_distance(face.normed_embedding, known_face["embedding"])
                        for known_face in self.window[self.id]
                    ]
                )
                if len(self.window[self.id]) > 0
                else float("inf")
            )
            face.dynamic_threshold = self.get_dynamic_threshold(face, height=height)
            ret.append(face)

        ret = sort_by_direction(
            ret,
            "distance-from-retarget-face"
            if direction == "distance-from-embedding" and len(self.window[self.id]) < self.window_size
            else direction,
            face_center,
            face_gender,
            # A number, not the word: with nothing averaged yet — a still frame
            # always starts there — this is multiplied by the face threshold.
            self.running_average[self.id] if self.running_average[self.id] else float("inf"),
        )
        if len(self.window[self.id]) == 0 and face_center[0] and face_center[1]:
            for face in ret:
                x1, y1, x2, y2 = face.bbox
                if x1 <= face_center[0] <= x2 and y1 <= face_center[1] <= y2:
                    ret = [face]
                    break
            else:
                ret = []

        if len(ret) > 0 and direction == "distance-from-embedding":
            self.update_memory(ret[0])
            if len(self.window[self.id]) > self.window_size - 1:
                if self.running_average[self.id] is None:
                    self.running_average[self.id] = max(
                        [
                            self.calculate_distance(ret[0].normed_embedding, known_face["embedding"])
                            for known_face in self.window[self.id]
                        ]
                    )
                else:
                    change = (
                        max(
                            [
                                self.calculate_distance(ret[0].normed_embedding, known_face["embedding"])
                                for known_face in self.window[self.id]
                            ]
                        )
                        - self.running_average[self.id]
                    )
                    change = np.clip(change, -self.max_rate_of_change, self.max_rate_of_change)
                    self.running_average[self.id] += change
            if len(self.window[self.id]) > self.max_window_size:
                self.window[self.id] = self.window[self.id][1:]
        return ret

    def warmup(self):
        self.timer.tic()
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)
        elapse = self.timer.toc()
        print(f"FaceAnalysis warmup time: {elapse:.3f}s")

    @staticmethod
    def calculate_distance(embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
