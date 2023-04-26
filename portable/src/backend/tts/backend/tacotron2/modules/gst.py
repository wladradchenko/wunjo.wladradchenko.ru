"""
Based on https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
"""
import torch
import torch.nn as nn

from modules.layers import LinearNorm, ConvBlock
from utils.utils import OutputsGST


class ReferenceEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        channels = zip([1] + hparams.reference_encoder_filters[:-1], hparams.reference_encoder_filters)

        self.convs = torch.nn.ModuleList(
            [ConvBlock(
                dimensions=2, in_channels=in_channels, out_channels=out_channels,
                kernel_size=hparams.reference_encoder_kernel, stride=hparams.reference_encoder_strides,
                padding=hparams.reference_encoder_pad, activation=hparams.reference_encoder_activation, bn=True,
                initscheme=hparams.initscheme, nonlinearity=hparams.reference_encoder_activation)
            for in_channels, out_channels in channels]
        )

        self.conv_params = {
            "kernel_size": hparams.reference_encoder_kernel[0],
            "stride": hparams.reference_encoder_strides[0],
            "pad": hparams.reference_encoder_pad[0],
            "n_convs": len(hparams.reference_encoder_filters)
        }

        self.n_mels = hparams.n_mel_channels

        out_channels = self.calculate_size(dim_size=self.n_mels, **self.conv_params)

        self.gru = torch.nn.GRU(
            input_size=hparams.reference_encoder_filters[-1] * out_channels,
            hidden_size=hparams.encoder_embedding_dim // 2,
            batch_first=True
        )


    def forward(self, inputs, input_lengths=None):
        assert inputs.size(-1) % self.n_mels == 0
        out = inputs.view(inputs.size(0), 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv in self.convs:
            out = conv(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            _input_lengths = self.calculate_size(input_lengths, **self.conv_params)
            out = nn.utils.rnn.pack_padded_sequence(
                out, _input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)


    @staticmethod
    def calculate_size(dim_size, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            dim_size = (dim_size - kernel_size + 2 * pad) // stride + 1
        return dim_size


class STL(torch.nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, hparams):
        super().__init__()

        self.embed = torch.nn.Parameter(torch.FloatTensor(
            hparams.stl_token_num, hparams.encoder_embedding_dim // hparams.stl_num_heads
        ))

        self.query_dim = hparams.encoder_embedding_dim // 2
        self.key_dim = hparams.encoder_embedding_dim // hparams.stl_num_heads

        self.attention = MultiHeadAttention(
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            dim=hparams.encoder_embedding_dim,
            heads=hparams.stl_num_heads
        )

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)


    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]

        weights, style_emb = self.attention(query, keys)
        weights = weights.squeeze(2)  # [N, num_heads, token_num]

        return weights, style_emb


class MultiHeadAttention(torch.nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, dim, heads):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.dim = dim
        self.heads = heads

        self.W_query = LinearNorm(query_dim, dim, bias=False)
        self.W_key = LinearNorm(key_dim, dim, bias=False)
        self.W_value = LinearNorm(key_dim, dim, bias=False)


    def forward(self, query, key):
        q = self.W_query(query)  # [b, t_q, dim]
        k = self.W_key(key)  # [b, t_k, dim]
        v = self.W_value(key)

        b, head_dim = query.size(0), self.dim // self.heads
        split_heads = lambda x: x.view(b, -1, self.heads, head_dim).transpose(2, 1)
        q, k, v = map(split_heads, (q, k, v))  # [b, h, t_q/k, head_dim]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(q, k.transpose(2, 3))  # [b, h, t_q, t_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = scores.softmax(dim=-1)

        # out = score * V
        out = torch.matmul(scores, v)  # [b, h, t_q, head_dim]
        out = out.transpose(2, 1).contiguous().view(b, -1, self.dim)  # [b, t_q, dim]

        return scores, out


class GST(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = ReferenceEncoder(hparams)
        self.stl = STL(hparams)

        self.device = torch.device("cpu" if not torch.cuda.is_available() else hparams.device)


    def forward(self, inputs, input_lengths=None):
        weights, style_emb = self.stl(self.encoder(inputs, input_lengths=input_lengths))

        outputs = OutputsGST(
            style_emb=style_emb,
            gst_weights=weights
        )

        return outputs


    def inference(self, encoder_outputs, reference_mel=None, token_idx=None):
        style_embedding = None
        if reference_mel is not None:
            _, style_embedding = self._forward(reference_mel)
            style_embedding = style_embedding.expand_as(encoder_outputs)
        elif token_idx is not None:
            query = self.stl.embed.new_zeros(1, 1, self.stl.query_dim)
            token = torch.tanh(self.stl.embed[token_idx]).view(1, 1, -1)
            _, style_embedding = self.stl.attention(query, token)

        return style_embedding