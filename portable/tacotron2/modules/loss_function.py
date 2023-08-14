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
import warnings
from collections import OrderedDict
from enum import Enum

import torch
from torch import nn

from utils.utils import get_mask_from_lengths


class AttentionTypes(str, Enum):
    none = "none"
    diagonal = "diagonal"
    prealigned = "prealigned"


    @staticmethod
    def guided_types():
        return [AttentionTypes.diagonal, AttentionTypes.prealigned]


class LossesType(str, Enum):
    MSE = "MSE"
    L1 = "L1"


mel_loss_func = {
    LossesType.MSE: nn.MSELoss,
    LossesType.L1: nn.L1Loss
}


def diagonal_guide(text_len, mel_len, g=0.2):
    grid_text = torch.linspace(0., 1. - 1. / text_len, text_len)  # (T)
    grid_mel = torch.linspace(0., 1. - 1. / mel_len, mel_len)  # (M)
    grid = grid_text.view(1, -1) - grid_mel.view(-1, 1)  # (M, T)

    W = 1 - torch.exp(-grid ** 2 / (2 * g ** 2))
    return W


def batch_diagonal_guide(text_lengths, mel_lengths, g=0.2):
    dtype, device = torch.float32, text_lengths.device

    grid_text = torch.arange(text_lengths.max(), dtype=dtype, device=device)
    grid_text = grid_text.view(1, -1) / text_lengths.view(-1, 1)  # (B, T)

    grid_mel = torch.arange(mel_lengths.max(), dtype=dtype, device=device)
    grid_mel = grid_mel.view(1, -1) / mel_lengths.view(-1, 1)  # (B, M)

    grid = grid_text.unsqueeze(1) - grid_mel.unsqueeze(2)  # (B, M, T)

    # apply text and mel length masks
    grid.transpose(2, 1)[~get_mask_from_lengths(text_lengths)] = 0.
    grid[~get_mask_from_lengths(mel_lengths)] = 0.

    W = 1 - torch.exp(-grid ** 2 / (2 * g ** 2))
    return W


def diagonal_loss(predicted, text_len, mel_len, g=0.2):
    guide = predicted.new_zeros(predicted.shape)
    guide[:mel_len, :text_len] = diagonal_guide(text_len, mel_len, g)
    return predicted * guide


def prealigned_loss(target, predicted):
    target = target if target.max != 0 else predicted # если матрица выравнивания недоступна - ошибка=0
    return (predicted - target) ** 2


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.mel_loss_type = LossesType(hparams.mel_loss_type)
        self.mel_criterion = mel_loss_func[self.mel_loss_type]()

        self.gate_positive_weight = torch.tensor(hparams.gate_positive_weight)
        self.gate_criterion = nn.BCEWithLogitsLoss(pos_weight=self.gate_positive_weight)


    def forward(self, mels_out, mels_postnet_out, gate_out, mels_target, gate_target):
        # ошибка мелов
        mels_target.requires_grad = False

        mel_loss = self.mel_criterion(mels_out, mels_target) + \
                   self.mel_criterion(mels_postnet_out, mels_target)

        # ошибка сигнала окончания предложения
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        gate_out = gate_out.view(-1, 1)

        gate_loss = self.gate_criterion(gate_out, gate_target)

        return mel_loss, gate_loss


class AttentionLoss(nn.Module):
    def __init__(self, hparams):
        self.type = AttentionTypes(hparams.guided_attention_type)
        self.diagonal_factor = None
        if self.type == AttentionTypes.diagonal:
            self.diagonal_factor = hparams.diagonal_factor
        self.attention_weight = hparams.attention_weight

        self.include_padding = hparams.include_padding

        super().__init__()


    def forward(self, attn_out, attn_target, text_lengths, mel_lengths):
        batch_size = len(text_lengths)
        attention_loss = 0

        if self.type == AttentionTypes.none:
            return
        else:
            if self.type == AttentionTypes.diagonal:
                diagonal_guides = batch_diagonal_guide(
                    text_lengths=text_lengths,
                    mel_lengths=mel_lengths,
                    g=self.diagonal_factor
                )
                attention_loss = torch.sum(attn_out * diagonal_guides)

            elif self.type == AttentionTypes.prealigned:
                for i in range(batch_size):
                    if attn_target[i].max() == 0.:
                        attn_target[i] = attn_out[i]  # если матрица выравнивания недоступна - ошибка=0

                attention_loss += nn.MSELoss(reduction="sum")(attn_out, attn_target)

            else:
                raise TypeError

            if not self.include_padding:
                active_elements = torch.sum(text_lengths * mel_lengths)
            else:
                active_elements = batch_size * text_lengths.max() * mel_lengths.max()

            attention_loss = attention_loss / active_elements * self.attention_weight

            return attention_loss


class OverallLoss(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.basic_criterion = Tacotron2Loss(hparams)
        self.list = ["overall/loss", "decoder/mel_loss", "decoder/gate_loss"]

        self.attention_criterion = None
        if hparams.guided_attention_type in AttentionTypes.guided_types():
            if hparams.guided_attention_type == AttentionTypes.prealigned:
                assert not hparams.word_level_prob
            self.attention_criterion = AttentionLoss(hparams)
            self.list.append("decoder/attention_loss")

        self.mmi_criterion = None
        if hparams.use_mmi:
            from modules.mmi import MIEsitmator # импортим тут, чтобы избежать циклических импортов
            self.mmi_criterion = MIEsitmator(hparams)
            self.list.append("mi/loss")


    def forward(self, outputs, inputs, **kwargs):
        losses = OrderedDict({loss: 0 for loss in self.list})

        mel_loss, gate_loss = self.basic_criterion(
            outputs.mels, outputs.mels_postnet, outputs.gate,
            inputs.mels, inputs.gate
        )
        losses["decoder/mel_loss"] = mel_loss
        losses["decoder/gate_loss"] = gate_loss
        losses["overall/loss"] += (mel_loss + gate_loss)

        if self.attention_criterion is not None:
            alignments = kwargs.get("alignments", None)
            if alignments is None and self.attention_criterion.type == AttentionTypes.prealigned:
                warnings.warn("Insufficient number of arguments to calculate the attention loss")
            else:
                attention_loss = self.attention_criterion(
                    outputs.alignments, alignments,
                    inputs.text_len, inputs.mel_len
                )
                losses["decoder/attention_loss"] = attention_loss
                losses["overall/loss"] += attention_loss

        if self.mmi_criterion is not None:
            inputs_ctc = kwargs.get("inputs_ctc", None)
            decoder_outputs = kwargs.get("decoder_outputs", None)
            if inputs_ctc is None or decoder_outputs is None:
                warnings.warn("Insufficient number of arguments to calculate the mi loss")
            else:
                mi_loss = self.mmi_criterion(
                    decoder_outputs, inputs_ctc.text,
                    inputs.mel_len, inputs_ctc.length
                )
                losses["mi/loss"] = mi_loss
                losses["overall/loss"] += mi_loss

        return losses