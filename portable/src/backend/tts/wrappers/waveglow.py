import sys
import torch
import numpy as np

from denoiser import Denoiser
import glow


_waveglow_path = sys.path[0]


class WaveglowWrapper:
    def __init__(self, model_path, device, sigma=0.666, strength=0.1):
        self.device = "cpu"
        self.dtype = torch.float

        self.model = torch.load(model_path, map_location=self.device)["model"]
        self.model.device = self.device

        for m in self.model.modules():
            if "Conv" in str(type(m)):
                setattr(m, "padding_mode", "zeros")

        self.model.eval().to(device=self.device, dtype=self.dtype)

        for k in self.model.convinv:
            k.float()

        self.denoiser = Denoiser(self.model, device=self.device)

        self.sigma = sigma
        self.strength = strength


    def __call__(self, spectrogram):
        with torch.no_grad():
            audio = self.model.infer(spectrogram, self.sigma)

        return audio


    def denoise(self, audio):
        if type(audio) == np.ndarray:
            audio = torch.tensor(audio).to(self.device, self.dtype)

        if audio.ndim == 1:
            audio = audio.view(1, -1)
        audio = self.denoiser(audio, self.strength)[:, 0]

        return audio.data.cpu().numpy()


    @staticmethod
    def clear_cache():
        torch.cuda.empty_cache()