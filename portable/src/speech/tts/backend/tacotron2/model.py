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
from math import sqrt

import torch
from numpy import finfo
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

root_path = os.path.dirname(os.path.abspath(__file__))
tps_path = os.path.dirname(os.path.dirname(os.path.dirname(root_path)))
sys.path.insert(0, os.path.join(tps_path, "tps"))

from tps import get_symbols_length, prob2bool

sys.path.pop(0)

from modules.layers import ConvNorm, ConvBlock, LinearNorm, activation_func
from modules.gst import GST
from utils.distributed import apply_gradient_allreduce
from utils import utils as utl

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim, initscheme="xavier_uniform"):
        super(LocationLayer, self).__init__()

        self.location_conv = ConvNorm(
            dimensions=1,
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=int((attention_kernel_size - 1) / 2),
            bias=False,
            stride=1,
            dilation=1,
            initscheme=initscheme
        )
        self.location_dense = LinearNorm(
            in_dim=attention_n_filters,
            out_dim=attention_dim,
            bias=False,
            initscheme=initscheme,
            nonlinearity='tanh'
        )


    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)

        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, initscheme="xavier_uniform"):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            in_dim=attention_rnn_dim,
            out_dim=attention_dim,
            bias=False,
            initscheme=initscheme,
            nonlinearity="tanh"
        )
        self.memory_layer = LinearNorm(
            in_dim=embedding_dim,
            out_dim=attention_dim,
            bias=False,
            initscheme=initscheme,
            nonlinearity="tanh"
        )
        self.v = LinearNorm(in_dim=attention_dim, out_dim=1, bias=False, initscheme=initscheme)
        self.location_layer = LocationLayer(
            attention_n_filters=attention_location_n_filters,
            attention_kernel_size=attention_location_kernel_size,
            attention_dim=attention_dim,
            initscheme=initscheme
        )
        self.score_mask_value = -float("inf")


    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))

        energies = energies.squeeze(-1)

        return energies


    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, initscheme='xavier_uniform', activation="relu"):
        super(Prenet, self).__init__()

        in_sizes = [in_dim] + sizes[:-1]

        layers = []
        for in_size, out_size in zip(in_sizes, sizes):
            layers.extend([
                LinearNorm(in_size, out_size, bias=False, initscheme=initscheme, nonlinearity="linear"),
                activation_func(activation)
            ])

        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(layer(x), p=0.5, training=True) if isinstance(layer, LinearNorm) else layer(x)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        params = [(hparams.postnet_embedding_dim, hparams.postnet_embedding_dim, "tanh")
                  for _ in range(hparams.postnet_n_convolutions)]

        params[0] = (hparams.n_mel_channels, hparams.postnet_embedding_dim, "tanh")
        params[-1] = (hparams.postnet_embedding_dim, hparams.n_mel_channels, "linear")

        _modules = [
            ConvBlock(
                dimensions=1,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=hparams.postnet_kernel_size,
                stride=1,
                padding=int((hparams.postnet_kernel_size - 1) / 2),
                dilation=1,
                activation=activation,
                bn=True,
                dropout=0.5,
                initscheme=hparams.initscheme,
                nonlinearity=activation
            ) for in_channels, out_channels, activation in params
        ]

        self.convolutions = nn.Sequential(*_modules)


    def forward(self, x):
        return self.convolutions(x)


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        _convolutions = [
            ConvBlock(
                dimensions=1,
                in_channels=hparams.encoder_embedding_dim,
                out_channels=hparams.encoder_embedding_dim,
                kernel_size=hparams.encoder_kernel_size,
                stride=1,
                padding=int((hparams.encoder_kernel_size - 1) / 2),
                dilation=1,
                activation=hparams.activation,
                bn=True,
                dropout=0.5,
                initscheme=hparams.initscheme,
                nonlinearity=hparams.activation
            )
            for _ in range(hparams.encoder_n_convolutions)
        ]

        self.convolutions = nn.Sequential(*_convolutions)

        self.lstm = nn.LSTM(
            input_size=hparams.encoder_embedding_dim,
            hidden_size=int(hparams.encoder_embedding_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )


    def forward(self, x, input_lengths):
        x = self.convolutions(x)
        x = x.transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


    def inference(self, x):
        x = self.convolutions(x)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold

        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.use_mmi = hparams.use_mmi

        self.prenet = Prenet(
            in_dim=hparams.n_mel_channels * hparams.n_frames_per_step,
            sizes=[hparams.prenet_dim, hparams.prenet_dim],
            initscheme=hparams.initscheme,
            activation=hparams.activation
        )

        self.attention_rnn = nn.LSTMCell(
            input_size=hparams.prenet_dim + hparams.encoder_embedding_dim,
            hidden_size=hparams.attention_rnn_dim
        )

        self.attention_layer = Attention(
            attention_rnn_dim=hparams.attention_rnn_dim,
            embedding_dim=hparams.encoder_embedding_dim,
            attention_dim=hparams.attention_dim,
            attention_location_n_filters=hparams.attention_location_n_filters,
            attention_location_kernel_size=hparams.attention_location_kernel_size,
            initscheme=hparams.initscheme
        )

        self.decoder_rnn = nn.LSTMCell(
            input_size=hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hidden_size=hparams.decoder_rnn_dim,
            bias=True
        )

        lp_out_dim = hparams.decoder_rnn_dim if self.use_mmi else hparams.n_mel_channels * hparams.n_frames_per_step

        self.mel_layer = None
        if not self.use_mmi:
            self.linear_projection = LinearNorm(
                in_dim=hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
                out_dim=lp_out_dim,
                bias=True,
                initscheme=hparams.initscheme
            )
        else:
            self.linear_projection = nn.Sequential(
                LinearNorm(
                    in_dim=hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
                    out_dim=lp_out_dim,
                    bias=True,
                    initscheme=hparams.initscheme,
                    nonlinearity="relu"
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            )

            self.mel_layer = nn.Sequential(
                LinearNorm(
                    in_dim=hparams.decoder_rnn_dim,
                    out_dim=hparams.decoder_rnn_dim,
                    bias=True,
                    initscheme=hparams.initscheme,
                    nonlinearity="relu"),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                LinearNorm(
                    in_dim=hparams.decoder_rnn_dim,
                    out_dim=hparams.n_mel_channels * hparams.n_frames_per_step)
            )

        gate_in_dim = hparams.decoder_rnn_dim if self.use_mmi else \
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim

        self.gate_layer = LinearNorm(
            in_dim=gate_in_dim,
            out_dim=1,
            bias=True,
            nonlinearity="sigmoid"
        )


    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input


    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask


    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs


    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, decoder_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        """
        # [(B, T_in) x T_out] -> (B, T_out, T_in)
        alignments = torch.stack(alignments).transpose(0, 1)

        # [(B, 1) x T_out] -> (B, T_out)
        gate_outputs = torch.cat(gate_outputs, dim=1).contiguous()

        # [(B, 1, n_mel_channels) x T_out] -> (B, T_out, n_mel_channels)
        mel_outputs = torch.cat(mel_outputs, dim=1)
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        if decoder_outputs:
            decoder_outputs = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
            decoder_outputs = decoder_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments, decoder_outputs


    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        if self.use_mmi:
            mel_output = self.mel_layer(decoder_output)
            decoder_hidden_attention_context = decoder_output
        else:
            mel_output = decoder_output
            decoder_output = None

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return mel_output, gate_prediction, self.attention_weights, decoder_output


    def forward(self, memory, decoder_inputs, memory_lengths, p_teacher_forcing=1.0):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~utl.get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments, decoder_outputs = [], [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if prob2bool(p_teacher_forcing) or len(mel_outputs) == 0:
                decoder_input = decoder_inputs[len(mel_outputs)]
            else:
                decoder_input = self.prenet(mel_outputs[-1])

            mel_output, gate_output, attention_weights, decoder_output = self.decode(decoder_input)

            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)

            if decoder_output is not None:
                decoder_outputs.append(decoder_output)

        mel_outputs, gate_outputs, alignments, decoder_outputs = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, decoder_outputs)

        return mel_outputs, gate_outputs, alignments, decoder_outputs



    def inference(self, memory, max_decoder_steps=None, suppress_gate=False):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        max_decoder_steps = max_decoder_steps or self.max_decoder_steps

        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments, decoder_outputs = [], [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment, decoder_output = self.decode(decoder_input)

            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(alignment)

            if decoder_output is not None:
                decoder_outputs.append(decoder_output)

            if not suppress_gate and torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == max_decoder_steps:
                # print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments, _ = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, decoder_outputs)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.device = torch.device("cpu" if not torch.cuda.is_available() else hparams.device)

        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.embedding = nn.Embedding(get_symbols_length(hparams.charset), hparams.symbols_embedding_dim)

        std = sqrt(2.0 / (get_symbols_length(hparams.charset) + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        self.tf_replacement = hparams.tf_replacement
        self.p_tf_train = hparams.p_tf_train
        self.p_tf_val = hparams.p_tf_val

        assert 0 <= self.p_tf_train <= 1.0
        assert 0 <= self.p_tf_val <= 1.0

        self.global_mean = None
        if self.tf_replacement == "global_mean":
            self.global_mean = utl.load_global_mean(hparams.global_mean_npy)

        self.gst = None
        if hparams.use_gst:
            self.gst = GST(hparams)

        self.to(self.device)


    def parse_batch(self, batch):
        inputs, alignments, inputs_ctc = batch

        inputs = utl.Inputs(
            text=utl.to_gpu(inputs.text).long(),
            mels=utl.to_gpu(inputs.mels).float(),
            gate=utl.to_gpu(inputs.gate).float(),
            text_len=utl.to_gpu(inputs.text_len).long(),
            mel_len=utl.to_gpu(inputs.mel_len).long()
        )

        if alignments is not None:
            alignments = utl.to_gpu(inputs.alignments).float()

        if inputs_ctc is not None:
            inputs_ctc = utl.InputsCTC(
                text=utl.to_gpu(inputs_ctc.text).long(),
                length=utl.to_gpu(inputs_ctc.length).long()
            )

        return inputs, alignments, inputs_ctc


    def parse_output(self, outputs, output_lengths=None, text_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~utl.get_mask_from_lengths(output_lengths)
            outputs.mels.data.masked_fill_(mask.unsqueeze(1), 0.0)
            outputs.mels_postnet.data.masked_fill_(mask.unsqueeze(1), 0.0)
            outputs.gate.data.masked_fill_(mask, 1e3)

            if text_lengths is not None:
                outputs.alignments.data.masked_fill_(~utl.get_mask_3d(output_lengths, text_lengths), 0.0)

        return outputs


    def mask_decoder_output(self, decoder_outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~utl.get_mask_from_lengths(output_lengths)
            decoder_outputs.data.masked_fill_(mask.unsqueeze(1), 0.0)

        return decoder_outputs


    def forward(self, inputs, **kwargs):
        text, mels, _, text_lengths, output_lengths = inputs

        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        if self.gst is not None:
            gst_outputs = self.gst(inputs=mels.transpose(2, 1), input_lengths=output_lengths)
            encoder_outputs += gst_outputs.style_emb.expand_as(encoder_outputs)

        p_teacher_forcing = 1.0
        if self.tf_replacement == "global_mean":
            drop_frame_rate = 1.0 - self.p_tf_train if self.training else 1.0 - self.p_tf_val
            mels = utl.dropout_frame(mels, self.global_mean, output_lengths, drop_frame_rate)
        elif self.tf_replacement == "decoder_output":
            p_teacher_forcing = self.p_tf_train if self.training else self.p_tf_val

        mel_outputs, gate_outputs, alignments, decoder_outputs = self.decoder(encoder_outputs, mels,
                                                                              memory_lengths=text_lengths,
                                                                              p_teacher_forcing=p_teacher_forcing)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = utl.Outputs(
            mels=mel_outputs,
            mels_postnet=mel_outputs_postnet,
            gate=gate_outputs,
            alignments=alignments
        )

        if len(decoder_outputs) != 0:
            decoder_outputs = self.mask_decoder_output(decoder_outputs, output_lengths)

        return self.parse_output(outputs, output_lengths, text_lengths), decoder_outputs


    def inference(self, inputs, **kwargs):
        max_decoder_steps = kwargs.get("max_decoder_steps", None)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.gst is not None:
            reference_mel = kwargs.pop("reference_mel", None)
            token_idx = kwargs.pop("token_idx", None)

            gst_output = self.gst.inference(encoder_outputs, reference_mel, token_idx)
            if gst_output is not None:
                encoder_outputs += gst_output

        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs, max_decoder_steps)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = utl.Outputs(
            mels=mel_outputs,
            mels_postnet=mel_outputs_postnet,
            gate=gate_outputs,
            alignments=alignments
        )

        return outputs


def load_model(hparams, distributed_run=False):
    model = Tacotron2(hparams)

    if hparams.fp16_run and hparams.device != "cpu":
        model.decoder.attention_layer.score_mask_value = finfo("float16").min

    if distributed_run and hparams.device != "cpu":
        model = apply_gradient_allreduce(model)

    return model


def convert_weights(checkpoint_path):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]

    for key in list(model_dict.keys()):
        item = model_dict.pop(key)
        if key.startswith(("encoder.convolutions", "postnet.convolutions", "decoder.prenet.layers.1")):
            key = key.split(".")

            if "decoder" in key:
                key[3] = "2"
            else:
                key.insert(3, "block")

            key = ".".join(key)
        model_dict[key] = item

    checkpoint_dict["state_dict"] = model_dict

    new_path, ext = os.path.splitext(checkpoint_path)
    new_path = new_path + "_converted" + ext
    torch.save(checkpoint_dict, new_path)