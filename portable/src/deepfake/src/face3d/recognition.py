import os
import sys
import numpy as np
import torch  # import torch first to use cuda for onnx, also need to install onnxruntime-gpu for this

from insightface.app.face_analysis import FaceAnalysis

import onnxruntime


class FaceRecognition:
    """ONNX Face Recognition (if I will use onnx models)"""
    def __init__(self, model_path):
        # use cpu as with cuda on onnx can be problem
        access_providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in access_providers:
            provider = ["CUDAExecutionProvider"] if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else ["CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]
        self.face_analyser = FaceAnalysis(name='buffalo_l', root=model_path, providers=provider)
        self.face_analyser.prepare(ctx_id=0)

        self.window = []
        self.window_size = 50  # last n distances for sliding window
        self.running_average = None
        self.max_rate_of_change = 0.05  # maximum allowed rate of change for running average
        self.fixed_threshold = 0.6  # initial fixed threshold

    def get_faces(self, frame):
        return self.face_analyser.get(frame)

    @staticmethod
    def calculate_distance(embedding1, embedding2):
        # Your distance calculation here
        return np.linalg.norm(embedding1 - embedding2)

    def is_similar_face(self, embedding, face_pred_list) -> bool:
        if not face_pred_list:
            return False

        distances = [self.calculate_distance(embedding, known_embedding) for known_embedding in face_pred_list]

        current_average = np.mean(distances)

        if self.running_average is None:
            if len(self.window) < self.window_size:
                self.window.append(current_average)
                return current_average < self.fixed_threshold

            self.running_average = np.mean(self.window)

        # Sliding window average
        if len(self.window) >= self.window_size:
            self.window.pop(0)
        self.window.append(current_average)

        new_average = np.mean(self.window)

        # Limit the rate of change for running_average
        change = new_average - self.running_average
        change = np.clip(change, -self.max_rate_of_change, self.max_rate_of_change)

        self.running_average += change

        dynamic_threshold = self.running_average * 0.95  # TODO Maybe did this control param from frontend
        # print(min(distances), dynamic_threshold)

        return min(distances) < dynamic_threshold
