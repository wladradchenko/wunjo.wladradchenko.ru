import os
import uuid
import torch
import librosa
import subprocess
import soundfile as sf
from openunmix import predict


class AudioSeparator:
    @staticmethod
    def _convert_to_wav(audio_path, output_path):
        wav_audio_path = os.path.join(output_path, str(uuid.uuid4()) + ".wav")
        cmd = f"ffmpeg -i {audio_path} {wav_audio_path}"
        # not silence run TODO remove
        # os.system(cmd)
        # silence run
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_audio_path

    def separate_audio(self, wav_audio_path, output_path, converted_wav=False, target_wav="vocals", device="cpu",
                       resample=True, compare_result=True):
        # target_wav is "vocals" or "residual"
        # Convert to WAV
        wav_audio_path = self._convert_to_wav(wav_audio_path, output_path) if converted_wav else wav_audio_path

        # Load the converted WAV file
        audio, rate = sf.read(wav_audio_path)
        audio_tensor = torch.tensor(audio).float()

        # Separate sources using Open-Unmix
        # TODO check what will not be problem with download model on windows
        estimates = predict.separate(
            audio_tensor,
            rate=rate,
            targets=['vocals'],
            residual=True,
            device=device,
        )

        wav_name = str(uuid.uuid4())

        # Save separated sources
        for target, estimate in estimates.items():
            output_audio = estimate.detach().cpu().numpy().squeeze().T
            output_filename = os.path.join(output_path, f"{wav_name}_{target}.wav")
            if target_wav == target:
                if resample:
                    # Resample to 16 kHz
                    output_audio = librosa.resample(output_audio.T, orig_sr=rate, target_sr=16000)
                    # Save file
                    sf.write(output_filename, output_audio.T, 16000)
                else:
                    sf.write(output_filename, output_audio, rate)
                print(f"Saved: {output_filename}")
                break

        if compare_result:
            duration_more = self.compare_audio_duration(wav_audio_path, os.path.join(output_path, f"{wav_name}_{target_wav}.wav"))
            if duration_more:
                return wav_audio_path
            else:
                return os.path.join(output_path, f"{wav_name}_{target_wav}.wav")

        return os.path.join(output_path, f"{wav_name}_{target_wav}.wav")

    @staticmethod
    def compare_audio_duration(original_audio_path, separated_audio_path):
        # Load the audios and get their durations
        y_original, sr_original = librosa.load(original_audio_path, sr=None)
        y_separated, sr_separated = librosa.load(separated_audio_path, sr=None)

        duration_original = librosa.get_duration(y=y_original, sr=sr_original)
        duration_separated = librosa.get_duration(y=y_separated, sr=sr_separated)

        # Compare durations
        if duration_separated > duration_original * 1.05:  # more than 5% from original
            print(
                f"Separated audio ({duration_separated:.2f} seconds) is longer than the original ({duration_original:.2f} seconds).")
            return True
        else:
            print(
                f"Separated audio ({duration_separated:.2f} seconds) is shorter or equal to the original ({duration_original:.2f} seconds).")
            return False

    @staticmethod
    def trim_silence(audio_path, output_path):
        y, sr = librosa.load(audio_path, sr=None)

        # Trim the silence from the start and end
        # `top_db` is the threshold in dB below which audio is considered silent
        y_trimmed, index = librosa.effects.trim(y, top_db=60)

        # Save the trimmed audio
        output_file = os.path.join(output_path, str(uuid.uuid4()) + ".wav")
        sf.write(output_file, y_trimmed, sr)
        return output_file