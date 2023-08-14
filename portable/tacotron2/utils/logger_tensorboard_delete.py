"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import random
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from tacotron2.utils.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from tacotron2.utils.plotting_utils import plot_gate_outputs_to_numpy
from tacotron2.utils.utils import to_numpy

sys.path.pop(0)


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)


    def _log_scalar_dict(self, scalar_dict, iteration, mode):
        for key, value in scalar_dict.items():
            key = "overall/{}_loss".format(mode) if key == "overall/loss" else key
            self.add_scalar(key, value, iteration)


    def log_training(self, losses_dict, grad_norm, learning_rate, duration, iteration):
        self._log_scalar_dict(losses_dict, iteration, "train")

        if grad_norm:
            self.add_scalar("training/grad_norm", grad_norm, iteration)
        self.add_scalar("training/learning_rate", learning_rate, iteration)
        self.add_scalar("training/duration", duration, iteration)


    def log_validation(self, losses_dict, model, target, prediction, iteration, target_alignments=None):
        self._log_scalar_dict(losses_dict, iteration, "val")

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment target and predicted, mel target and predicted, gate target and predicted
        log_image = partial(self.add_image, global_step=iteration, dataformats="HWC")
        idx = random.randint(0, prediction.alignments.size(0) - 1)

        if target_alignments is not None:
            log_image("alignment/target", plot_alignment_to_numpy(to_numpy(target_alignments[idx]).T))

        log_image("alignment/predicted", plot_alignment_to_numpy(to_numpy(prediction.alignments[idx].T)))
        log_image("mel/target", plot_spectrogram_to_numpy(to_numpy(target.mels[idx])))
        log_image("mel/predicted", plot_spectrogram_to_numpy(to_numpy(prediction.mels[idx])))
        log_image("gate", plot_gate_outputs_to_numpy(
            to_numpy(target.gate[idx]), to_numpy(torch.sigmoid(prediction.gate[idx]))
        ))