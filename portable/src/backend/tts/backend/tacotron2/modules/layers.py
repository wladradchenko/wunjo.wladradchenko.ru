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
import torch
from librosa.filters import mel as librosa_mel_fn

from modules.stft import STFT
from utils.distributions import init_weights, DistTypes
from utils.audio_processing import dynamic_range_compression, dynamic_range_decompression


act_types = torch.nn.ModuleDict([
        ("linear", torch.nn.Identity()),
        ("relu", torch.nn.ReLU(inplace=True)),
        ("leaky_relu", torch.nn.LeakyReLU(inplace=True)),
        ("selu", torch.nn.SELU(inplace=True)),
        ("tanh", torch.nn.Tanh())
    ])


conv_types = {
    1: torch.nn.Conv1d,
    2: torch.nn.Conv2d
}


bn_types = {
    1: torch.nn.BatchNorm1d,
    2: torch.nn.BatchNorm2d
}


dropout_types = {
    1: torch.nn.Dropout,
    2: torch.nn.Dropout2d
}


def activation_func(act_type):
    return act_types[act_type]


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, initscheme="xavier_uniform", nonlinearity="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        init_weights(self.linear_layer.weight, DistTypes[initscheme], nonlinearity)

    def forward(self, x):
        return self.linear_layer(x)


class ConvBlock(torch.nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1,
                 bias=True, activation="relu", bn=True, dropout=None,
                 initscheme="xavier_uniform", nonlinearity="linear"):
        super().__init__()

        _modules = [
            ConvNorm(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(dilation * (kernel_size - 1) / 2) if padding is None else padding,
                dilation=dilation,
                bias=bias,
                initscheme=initscheme,
                nonlinearity=nonlinearity
            )
        ]

        if bn:
            _modules.append(bn_types[dimensions](num_features=out_channels))
        _modules.append(activation_func(activation))

        if dropout is not None:
            _modules.append(dropout_types[dimensions](p=dropout))

        self.block = torch.nn.Sequential(*_modules)


    def forward(self, data):
        return self.block(data)


class ConvNorm(torch.nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, initscheme="xavier_uniform", nonlinearity="linear"):
        super(ConvNorm, self).__init__()
        if dimensions == 1 and padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = conv_types[dimensions](in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, dilation=dilation,
                                           bias=bias)

        init_weights(self.conv.weight, DistTypes[initscheme], nonlinearity)


    def forward(self, signal):
        return self.conv(signal)


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)


    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output


    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output


    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
