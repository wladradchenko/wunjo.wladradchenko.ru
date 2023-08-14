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
from itertools import chain

import numpy as np
import librosa
import torch
import torch.utils.data
from scipy.io.wavfile import read


from speech.tps.tps import prob2bool, symbols, cleaners

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from tacotron2.modules import layers
from tacotron2.utils.utils import load_filepaths_and_text, Inputs, InputsCTC
from tacotron2.modules.loss_function import AttentionTypes

sys.path.pop(0)


ctc_mapping = {
    symbols.Charset.en: symbols.en.EN_SET,
    symbols.Charset.ru: symbols.ru.RU_SET,
}


def get_ctc_symbols(charset):
    return ["pad"] + ctc_mapping[charset] + ["blank"]


def get_ctc_symbols_length(charset):
    charset = symbols.Charset[charset]
    return len(get_ctc_symbols(charset))


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, text_handler, filelist_path, hparams):
        self.text_handler = text_handler

        self.data = load_filepaths_and_text(filelist_path)
        self.audio_path = hparams.audios_path
        self.alignment_path = hparams.alignments_path

        self.add_silence = hparams.add_silence
        self.hop_length = hparams.hop_length
        self.ft_window = hparams.filter_length
        self.trim_silence = hparams.trim_silence
        self.trim_top_db = hparams.trim_top_db

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk

        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.word_level_prob = hparams.word_level_prob
        self.mask_stress = hparams.mask_stress
        self.mask_phonemes = hparams.mask_phonemes

        self.get_alignments = hparams.guided_attention_type == AttentionTypes.prealigned
        if self.get_alignments:
            assert not self.word_level_prob and not self.add_silence

        self.use_mmi = hparams.use_mmi
        self.ctc_symbol_to_id = None
        if hparams.use_mmi:
            self.ctc_symbol_to_id = {s: i for i, s in enumerate(get_ctc_symbols(hparams.charset))}


    def __getitem__(self, index):
        if isinstance(index, slice):
            return (self.get_data(data) for data in self.data[index])
        else:
            return self.get_data(self.data[index])


    def __len__(self):
        return len(self.data)


    def _prob2bool(self, prob):
        return prob2bool(prob) if not self.word_level_prob else prob


    def get_data(self, sample):
        mask_stress =  self._prob2bool(self.mask_stress)
        mask_phonemes = self._prob2bool(self.mask_phonemes)

        audio_name, text = sample

        sequence = self.get_text(text, mask_stress, mask_phonemes)
        mel = self.get_mel(audio_name)

        alignment = None
        if self.get_alignments:
            target_shape = (mel.size(1), sequence.size(0))
            alignment = self.get_alignment(audio_name, mask_stress, mask_phonemes, target_shape)

        ctc_sequence = None
        if self.use_mmi:
            ctc_sequence = self.get_ctc_text(sequence.data.cpu().numpy())

        return sequence, mel, alignment, ctc_sequence


    def get_text(self, text, mask_stress, mask_phonemes):
        preprocessed_text = self.text_handler.process_text(
            text, cleaners.light_punctuation_cleaners, None, False,
            mask_stress=mask_stress, mask_phonemes=mask_phonemes
        )
        preprocessed_text = self.text_handler.check_eos(preprocessed_text)
        text_vector = self.text_handler.text2vec(preprocessed_text)

        text_tensor = torch.IntTensor(text_vector)
        return text_tensor


    def get_audio(self, filename, trim_silence=False, add_silence=False):
        filepath = os.path.join(self.audio_path, filename)

        sample_rate, audio = read(filepath)
        audio = np.float32(audio / self.max_wav_value) # faster than loading using librosa

        if sample_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sample_rate, self.sampling_rate))

        audio_ = audio.copy()

        if trim_silence:
            idxs = librosa.effects.split(
                audio_,
                top_db=self.trim_top_db,
                frame_length=self.ft_window,
                hop_length=self.hop_length
            )

            audio_ = np.concatenate([audio_[start:end] for start, end in idxs])

        if add_silence:
            audio_ = np.append(audio_, np.zeros(5 * self.hop_length))

        audio_ = torch.FloatTensor(audio_.astype(np.float32))
        audio_ = audio_.unsqueeze(0)
        audio_ = torch.autograd.Variable(audio_, requires_grad=False)

        return audio_


    def get_mel_from_audio(self, audio):
        melspec = self.stft.mel_spectrogram(audio)
        return torch.squeeze(melspec, 0)


    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio = self.get_audio(filename, self.trim_silence, self.add_silence)
            melspec = self.stft.mel_spectrogram(audio)
            melspec = torch.squeeze(melspec, 0)
        else:
            filepath = os.path.join(self.audio_path, filename)
            melspec = torch.from_numpy(np.load(filepath))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec


    def get_alignment(self, audio_name, mask_stress, mask_phonemes, target_shape):
        audio_name, _ = os.path.splitext(audio_name)
        alignment_name = audio_name + ".npy"

        if not mask_phonemes:
            alignment = None # у нас пока нет выравниваний для фонемного представления
        else:
            sub_dir = "original" if mask_stress else "stressed"
            filepath = os.path.join(self.alignment_path[sub_dir], alignment_name)

            alignment = np.load(filepath)

        # TODO: поправить эту хрень с alignment
        if alignment is None or alignment.shape != target_shape:
            print("Some problems with {}: expected {} shape, got {}".format(audio_name, target_shape, alignment.shape))
            alignment = np.zeros(shape=target_shape)

        alignment = torch.FloatTensor(alignment)

        return alignment


    def get_ctc_text(self, sequence):
        text = [self.text_handler.id_to_symbol[s] for s in sequence]
        return torch.IntTensor([self.ctc_symbol_to_id[s] for s in text if s in self.ctc_symbol_to_id])


class TextMelCollate:
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step


    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        get_alignment = not any(elem[2] is None for elem in batch)
        get_ctc_text = not any(elem[3] is None for elem in batch)

        batchsize = len(batch)

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        max_target_len = max([x[1].size(1) for x in batch])
        num_mels = batch[0][1].size(0)

        text_padded = torch.LongTensor(batchsize, max_input_len)
        text_padded.zero_()

        mel_padded = torch.FloatTensor(batchsize, num_mels, max_target_len)
        mel_padded.zero_()

        gate_padded = torch.FloatTensor(batchsize, max_target_len)
        gate_padded.zero_()

        alignments_padded = None
        if get_alignment:
            alignments_padded = torch.FloatTensor(batchsize, max_target_len, max_input_len)
            alignments_padded.zero_()

        ctc_text_padded = None
        ctc_text_lengths = None
        if get_ctc_text:
            max_ctc_text_len = max([len(x[3]) for x in batch])

            ctc_text_padded = torch.LongTensor(batchsize, max_ctc_text_len)
            ctc_text_padded.zero_()

            ctc_text_lengths = torch.LongTensor(batchsize)

        output_lengths = torch.LongTensor(batchsize)

        for i, idx in enumerate(ids_sorted_decreasing):
            text, mel, alignment, ctc_text = batch[idx]

            in_len = text.size(0)
            target_len = mel.size(1)
            output_lengths[i] = target_len

            text_padded[i, :in_len] = text
            mel_padded[i, :, :target_len] = mel
            gate_padded[i, target_len - 1:] = 1

            if get_alignment:
                alignments_padded[i, :target_len, :in_len] = alignment

            if get_ctc_text:
                ctc_txt_len = ctc_text.size(0)
                ctc_text_lengths[i] = ctc_txt_len

                ctc_text_padded[i, :ctc_txt_len] = ctc_text

        # # Right zero-pad mel-spec
        # if max_target_len % self.n_frames_per_step != 0:
        #     max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
        #     assert max_target_len % self.n_frames_per_step == 0

        inputs = Inputs(text=text_padded, mels=mel_padded, gate=gate_padded,
                        text_len=input_lengths, mel_len=output_lengths)

        inputs_ctc = InputsCTC(text=ctc_text_padded, length=ctc_text_lengths) if get_ctc_text else None

        return inputs, alignments_padded, inputs_ctc


class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batchsize, shuffle=False, optimize=False, len_diff=10):
        idxs = tuple(range(len(data_source.data)))

        self.optimize = optimize
        self.shuffle = shuffle
        self.batchsize = batchsize
        self.optimized_idxs = []

        if self.optimize:
            text_lengths = tuple(len(elem[1]) for elem in data_source.data)
            lengths_idxs_pairs = tuple(zip(text_lengths, idxs))

            lengths_idxs_pairs = sorted(lengths_idxs_pairs, key=lambda elem: elem[0])

            min_length = lengths_idxs_pairs[0][0]

            len_idxs = []
            min_len = min_length
            max_len = min_len + len_diff
            for j, (length, idx) in enumerate(lengths_idxs_pairs):
                if min_len <= length < max_len:
                    len_idxs.append(idx)
                    if j + 1 == len(lengths_idxs_pairs) and len_idxs:
                        self.optimized_idxs.append(len_idxs)
                else:
                    self.optimized_idxs.append(len_idxs)
                    len_idxs = [idx]
                    min_len = length
                    max_len = min_len + len_diff

            idxs = tuple(chain(*self.optimized_idxs))

        self.idxs = idxs

        if self.shuffle:
            self.reshuffle()


    def __iter__(self):
        for i in self.idxs:
            yield i

        if self.shuffle:
            self.reshuffle()


    def __len__(self):
        return len(self.idxs)


    def reshuffle(self):
        def _torch_shuffle(iterable):
            return tuple(iterable[i] for i in torch.randperm(len(iterable)).tolist())

        idxs = tuple(_torch_shuffle(elem) for elem in self.optimized_idxs) if self.optimize else self.idxs
        idxs = _torch_shuffle(idxs)

        if self.optimize:
            idxs = list(chain(*idxs))

            batches = len(idxs) // self.batchsize
            idxs = tuple(idxs[i * self.batchsize:(i + 1) * self.batchsize] for i in range(batches))
            idxs = _torch_shuffle(idxs)

            idxs = list(chain(*idxs))

        self.idxs = idxs