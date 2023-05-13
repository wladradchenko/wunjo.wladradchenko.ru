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
import numpy as np
from collections import namedtuple

import torch

Inputs = namedtuple("Inputs", ["text", "mels", "gate", "text_len", "mel_len"])

InputsCTC = namedtuple("InputsCTC", ["text", "length"])

Outputs = namedtuple("Outputs", ["mels", "mels_postnet", "gate", "alignments"])
OutputsGST = namedtuple("OutputsGST", ["style_emb", "gst_weights"])


def calculate_global_mean(data_loader, path=None):
    """
    Based on https://github.com/bfs18/tacotron2
    """
    sums = []
    frames = []
    print("Calculating global mean...")
    for i, batch in enumerate(data_loader):
        print("\rProcessing batch #{} out of {}".format(i + 1, len(data_loader)), end="")
        inputs, *_ = batch
        # padded values are 0.
        sums.append(inputs.mels.double().sum(dim=(0, 2)))
        frames.append(inputs.mel_len.double().sum())

    global_mean = (sum(sums) / sum(frames)).float()

    if path is not None:
        np.save(path, global_mean.numpy())

    return to_gpu(global_mean)


def load_global_mean(path):
    assert os.path.exists(path)
    global_mean = np.load(path)

    return to_gpu(torch.tensor(global_mean))


def get_mask_from_lengths(lengths):
    max_len = lengths.max()
    ids = torch.arange(max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask


def get_mask_3d(widths, heights):
    mask_width = get_mask_from_lengths(widths)
    mask_height = get_mask_from_lengths(heights)
    mask_3d = mask_width.unsqueeze(2) & mask_height.unsqueeze(1)
    return mask_3d


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    """
    Based on https://github.com/bfs18/tacotron2
    """
    batch_size = lengths.size(0)
    max_len = torch.max(lengths).item()
    mask = get_mask_from_lengths(lengths).float()
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask.float() * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate):
    """
    Based on https://github.com/bfs18/tacotron2
    """
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate)
    dropped_mels = (mels * (1.0 - drop_mask).unsqueeze(1) +
                    global_mean[None, :, None] * drop_mask.unsqueeze(1))
    return dropped_mels


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def to_numpy(tensor):
    return tensor.data.cpu().numpy()