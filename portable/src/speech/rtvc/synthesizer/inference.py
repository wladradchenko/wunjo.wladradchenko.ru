import os
import sys
import torch
import soundfile

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.synthesizer import audio
from speech.rtvc.synthesizer.hparams import hparams
from speech.rtvc.synthesizer.models.tacotron import Tacotron
from speech.rtvc.encoder.audio import preprocess_wav
from speech.tps.tps import Handler, ssml
from speech.tts.utils.async_utils import BackgroundGenerator
from speech.rtvc.synthesizer.utils.text import text_to_sequence

sys.path.pop(0)

from pathlib import Path
from typing import Union, List
import numpy as np
import librosa


class Synthesizer:
    hparams = hparams
    sample_rate = hparams.sample_rate

    def __init__(self, model_fpath: Path, verbose=True, device="cpu", pause_type="silence", charset="en"):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.

        :param model_fpath: path to the trained model file
        :param verbose: if False, prints less information when using the model
        :param device: device
        """
        self.model_fpath = model_fpath
        self.verbose = verbose
        self.device = torch.device(device)  # set device from choose user

        if self.verbose:
            print("Synthesizer using device:", self.device)

        # Tacotron model will be instantiated later on first use.
        self._model = None

        self.pause_type = pause_type
        self.text_handler = self._load_text_handler({
            "config": charset,
            "out_max_length": 200
        }, use_cleaner=False)

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._model is not None

    def load(self, symbols):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        """
        self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
                               num_chars=len(symbols),
                               encoder_dims=hparams.tts_encoder_dims,
                               decoder_dims=hparams.tts_decoder_dims,
                               n_mels=hparams.num_mels,
                               fft_bins=hparams.num_mels,
                               postnet_dims=hparams.tts_postnet_dims,
                               encoder_K=hparams.tts_encoder_K,
                               lstm_dims=hparams.tts_lstm_dims,
                               postnet_K=hparams.tts_postnet_K,
                               num_highways=hparams.tts_num_highways,
                               dropout=hparams.tts_dropout,
                               stop_threshold=hparams.tts_stop_threshold,
                               speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)

        self._model.load(self.model_fpath)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer \"%s\" trained to step %d" % (self.model_fpath, self._model.state_dict()["step"]))

    def _sequence_audio_gen(self, sequence_generator, embeddings):
        # Mappings from symbol to numeric ID and vice versa:
        _symbol_to_id = {s: i for i, s in enumerate(self.text_handler.voice_clone_symbols)}

        spectrograms = []
        for unit in sequence_generator:
            if isinstance(unit, ssml.Pause):
                continue
            else:
                unit_value = self.text_handler.check_eos(unit.value)
                phrases = self.split_into_phrases(unit_value)
                for phrase in phrases:
                    print(phrase)
                    phrase = [text_to_sequence(phrase.replace("~", ""), hparams.tts_cleaner_names, _symbol_to_id)]
                    batched_inputs = [phrase[i:i + hparams.synthesis_batch_size] for i in range(0, len(phrase), hparams.synthesis_batch_size)]
                    batched_embeds = [embeddings[i:i + hparams.synthesis_batch_size] for i in range(0, len(embeddings), hparams.synthesis_batch_size)]
                    for i, batch in enumerate(batched_inputs, 1):
                        if self.verbose:
                            print(f"\n| Generating {i}/{len(batched_inputs)}")

                        # Pad texts so they are all the same length
                        text_lens = [len(text) for text in batch]
                        max_text_len = max(text_lens)
                        chars = [self.pad1d(text, max_text_len) for text in batch]
                        chars = np.stack(chars)

                        # Stack speaker embeddings into 2D array for batch processing
                        speaker_embeds = np.stack(batched_embeds[i - 1])

                        # Convert to tensor
                        chars = torch.tensor(chars).long().to(self.device)
                        speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

                        # Inference
                        _, mels, alignments = self._model.generate(chars, speaker_embeddings)
                        mels = mels.detach().cpu().numpy()
                        for m in mels:
                            # Trim silence from end of each spectrogram
                            while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                                m = m[:, :-1]
                            spectrograms.append(m)
        return spectrograms

    def synthesize(self,vocoder, text: str, embeddings: Union[np.ndarray, List[np.ndarray]], return_alignments=False):
        audio_list = self.synthesize_spectrograms(text=text, embeddings=embeddings)
        for audio in audio_list:
            yield vocoder.infer_waveform(audio)


    def save(self, audio, path, name, prefix=None):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, name)
        soundfile.write(file_path, audio.astype(np.float32), self.sample_rate)

        return file_path

    def synthesize_spectrograms(self, text: str, embeddings: Union[np.ndarray, List[np.ndarray]], return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param text: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param return_alignments: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        print(self.text_handler.language)
        # Load the model on the first request.
        if not self.is_loaded():
            self.load(self.text_handler.voice_clone_symbols)

        # Add auto transcript a text
        text = text.lower()
        # if self.text_handler.language == "chinese":
        #     text = " ".join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))

        cleaners = ("light_punctuation_cleaners",)

        if text.startswith("<speak>") and text.endswith("</speak>"):
            sequence = ssml.parse_ssml_text(text)
        else:
            sequence = self.text_handler.split_to_sentences(text, True, self.text_handler.language)
            sequence = [
                ssml.Text(elem) if not isinstance(elem, ssml.Pause) else elem for elem in sequence
            ]

        sequence_generator = BackgroundGenerator(
            self._sequence_to_sequence_gen(sequence, cleaners, None, None)
        )

        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        return self._sequence_audio_gen(sequence_generator, embeddings)

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer.
        """
        wav = librosa.load(str(fpath), hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav

        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)

    @staticmethod
    def pad1d(x, max_len, pad_value=0):
        return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

    @staticmethod
    def _load_text_handler(config, use_cleaner = True):
        out_max_length = config["out_max_length"]
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tps",  "rules")
        handler = Handler.from_charset(config["config"], data_dir=config_path, out_max_length=out_max_length, silent=True, use_cleaner=use_cleaner)

        return handler

    def _sequence_to_sequence_gen(self, sequence, cleaners, mask_stress, mask_phonemes):
        for element in sequence:
            if not isinstance(element, ssml.Pause):
                sentence = element.value

                sentence = self.text_handler.process(
                    string=sentence,
                    cleaners=cleaners,
                    user_dict=None,
                    mask_stress=mask_stress,
                    mask_phonemes=mask_phonemes
                )

                if self.text_handler.out_max_length is not None:
                    _units = self.text_handler.split_to_units(sentence, self.text_handler.out_max_length, True)

                    for unit in _units:
                        yield ssml.Text(unit).inherit(element) if not isinstance(unit, ssml.Pause) else unit
                else:
                    element.update_value(sentence)
                    yield element
            else:
                yield element


    @staticmethod
    def split_into_phrases(text: str, max_words: int = 15, min_words: int = 10) -> list:
        """
        Split big phrases and phrase not more 20 words
        :param text: text
        :param max_words: max words
        :param min_words: min words after those if will be some symbol, when create phrase
        :return: phrases
        """
        words = text.replace("\n", " ").replace("\t", " ").split(" ")
        phrases = []
        phrase = []
        word_count = 0

        for word in words:
            if word_count + len(word.split()) <= max_words:
                phrase.append(word)
                word_count += len(word.split())
                # Check for the special condition
                if word_count > min_words and any(word.endswith(char) for char in ['.', ',', '%', '!', '?']):
                    strip_phrase = " ".join(phrase).strip()
                    if strip_phrase:
                        phrases.append(strip_phrase)
                    phrase = []
                    word_count = 0
            else:
                strip_phrase = " ".join(phrase).strip()
                if strip_phrase:
                    phrases.append(strip_phrase)
                phrase = [word]
                word_count = len(word.split())

        strip_phrase = " ".join(phrase).strip()
        if strip_phrase:
            phrases.append(strip_phrase)

        return phrases