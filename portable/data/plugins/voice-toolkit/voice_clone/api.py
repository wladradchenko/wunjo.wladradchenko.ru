"""The converter itself and the mark it leaves on what it produces.

Ported from Wunjo v2 (portable/src/sound_processing/clone_voice/api.py). Two
things came across: the OpenVoice tone-colour converter — the network that keeps
what was said and replaces who says it — and the digital signature the old app
wrote into every voice it made, so that Wunjo can still tell its own output from
a recording.

What did not come across is the file-by-file work the old API did around them
(loading a track, writing a wav, cutting the source into pieces on disk). A
clip's sound arrives here as an array and leaves as one; ``clone.py`` owns that
side, which is what lets a long clip be converted in pieces without ever writing
a temporary file per piece.
"""
import os

import librosa
import numpy as np
import torch

from . import utils
from .models import SynthesizerTrn
from .signature import load_model


class OpenVoiceBaseClass(object):
    def __init__(self,
                 config_path,
                 device='cuda:0'):
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = utils.get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, 'symbols', [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device), weights_only=True)
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print('missing/unexpected keys:', a, b)


class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = getattr(self.hps, '_version_', "v1")

    @property
    def sampling_rate(self) -> int:
        return int(self.hps.data.sampling_rate)


class DigitalSignature:
    """The mark Wunjo writes into a voice it made.

    It is inaudible and it is not protection — anyone who re-encodes the file
    hard enough will lose it. It exists so that a track which came out of here
    can be recognised as such later, which is the least a tool that copies a
    voice owes to the people who listen to the result.
    """

    def __init__(self, device):
        self.device = device
        model_path = os.path.join(os.path.dirname(__file__), "signature.pkl")
        self.model = load_model(model_path).to(self.device)

    def set_encrypted(self, audio, message="ai"):
        device = self.device
        bits = utils.string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to add signature')
                break
            message_npy = bits[n * 32: (n + 1) * 32]

            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = self.model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K: (coeff * n + 1) * K] = signal_wmd_npy

        return audio

    def decrypted(self, audio_path, message="ai") -> bool:
        # Load audio data from the file
        audio, sample_rate = librosa.load(audio_path, sr=None)

        # Ensure the audio is in the correct format
        if audio.ndim > 1:  # Convert stereo to mono if needed
            audio = audio.mean(axis=1)

        bits = utils.string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        bits = []
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to detect signature')
                return False
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device).unsqueeze(0)
                message_decoded_npy = (self.model.decode(signal) >= 0.5).int().detach().cpu().numpy().squeeze()
            bits.append(message_decoded_npy)
        bits = np.stack(bits).reshape(-1, 8)
        decoded_message = utils.bits_to_string(bits).strip()
        return decoded_message != message
