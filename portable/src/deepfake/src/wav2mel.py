import os
import subprocess
import numpy as np
import src.utils.audio as audio

class MelProcessor:
    def __init__(self, audio, save_output, fps):
        self.audio = audio
        self.save_output = save_output
        self.fps = fps
        self.mel_step_size = 16

    def convert_to_wav(self):
        if not self.audio.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(self.audio, os.path.join(self.save_output, "transfer_audio_temp.wav"))
            subprocess.call(command, shell=True)
            if os.environ.get('DEBUG', 'False') == 'True':
                # not silence run
                subprocess.call(command, shell=True)
            else:
                # silence run
                subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.audio = os.path.join(self.save_output, "transfer_audio_temp.wav")

    def load_audio(self):
        wav = audio.load_wav(self.audio, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)
        return mel

    def check_for_nan(self, mel):
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    def chunk_mel(self, mel, fps, mel_step_size):
        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))
        return mel_chunks

    def process(self):
        self.convert_to_wav()
        mel = self.load_audio()
        self.check_for_nan(mel)
        mel_chunks = self.chunk_mel(mel, self.fps, self.mel_step_size)
        return mel_chunks