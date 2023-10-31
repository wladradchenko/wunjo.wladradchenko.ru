import os
import sys
import numpy as np
import noisereduce as nr
from denoiser.pretrained import master64

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.vocoder.models.fatchord_version import WaveRNN
from speech.rtvc.vocoder import hparams as hp

sys.path.pop(0)

import torch


class VoiceCloneVocoder:
    """
    VoiceCloneVocoder
    """
    _model = None   # type: WaveRNN

    def load_model(self, weights_fpath, verbose=True, device="cpu"):
        if verbose:
            print("Building Wave-RNN")
        _model = WaveRNN(
            rnn_dims=hp.voc_rnn_dims,
            fc_dims=hp.voc_fc_dims,
            bits=hp.bits,
            pad=hp.voc_pad,
            upsample_factors=hp.voc_upsample_factors,
            feat_dims=hp.num_mels,
            compute_dims=hp.voc_compute_dims,
            res_out_dims=hp.voc_res_out_dims,
            res_blocks=hp.voc_res_blocks,
            hop_length=hp.hop_length,
            sample_rate=hp.sample_rate,
            mode=hp.voc_mode
        )

        if torch.cuda.is_available() and device == "cuda":
            self._model = _model.cuda()
            self._device = torch.device('cuda')
        else:
            self._model = _model.cpu()
            self._device = torch.device('cpu')

        if verbose:
            print("Loading model weights at %s" % weights_fpath)
        checkpoint = torch.load(weights_fpath, self._device)
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()

    def is_loaded(self):
        return self._model is not None

    def infer_waveform(self, mel, normalize=True,  batched=True, target=8000, overlap=800, progress_callback=None, device="cpu"):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer (the format must match
        that of the synthesizer!)

        :param normalize:
        :param batched:
        :param target:
        :param overlap:
        :return:
        """
        if self._model is None:
            raise Exception("Please load Wave-RNN in memory before using it")

        if normalize:
            mel = mel / hp.mel_max_abs_value
        mel = torch.from_numpy(mel[None, ...])
        wav = self._model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
        wav = self.waveform_denoising(wav, device)
        return wav

    @staticmethod
    def waveform_denoising(wav, device):
        prop_decrease = hp.prop_decrease_low_freq
        # TODO if on Windows will be problem to download, use location .wunjo and download by request
        model = master64().to(device)
        noisy = torch.from_numpy(np.array([wav])).to(device).float()
        estimate = model(noisy)
        estimate = estimate * (1 - hp.dry) + noisy * hp.dry
        estimate = estimate[0].cpu().detach().numpy()
        return nr.reduce_noise(np.squeeze(estimate), hp.sample_rate, prop_decrease=prop_decrease)
