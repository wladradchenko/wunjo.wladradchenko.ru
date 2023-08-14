"""
Based on https://github.com/bfs18/tacotron2
"""
import torch

from modules.layers import LinearNorm
from utils.data_utils import get_ctc_symbols_length


class MIEsitmator(torch.nn.Module):
    def __init__(self, hparams, dropout=0.5):
        super().__init__()

        self.device = torch.device("cpu" if not torch.cuda.is_available() else hparams.device)

        vocab_size = get_ctc_symbols_length(hparams.charset)
        decoder_dim = hparams.decoder_rnn_dim

        self.use_gaf = hparams.use_gaf

        self.proj = torch.nn.Sequential(
            LinearNorm(decoder_dim, decoder_dim, bias=True, initscheme="xavier_uniform", nonlinearity="relu"),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout)
        )
        self.ctc_proj = LinearNorm(decoder_dim, vocab_size, bias=True)
        self.ctc = torch.nn.CTCLoss(blank=vocab_size - 1, reduction="none", zero_infinity=True)

        self.to(self.device)


    def forward(self, decoder_outputs, target_text, decoder_lengths, target_lengths):
        # transpose to [b, T, dim]
        decoder_outputs = decoder_outputs.transpose(2, 1)

        out = self.proj(decoder_outputs)

        log_probs = self.ctc_proj(out).log_softmax(dim=2)
        log_probs = log_probs.transpose(1, 0)

        ctc_loss = self.ctc(log_probs, target_text, decoder_lengths, target_lengths)
        # average by number of frames since taco_loss is averaged.
        ctc_loss = (ctc_loss / decoder_lengths.float()).mean()

        return ctc_loss