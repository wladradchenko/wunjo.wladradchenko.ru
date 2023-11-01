import os
import sys
import uuid
from time import time
import subprocess

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from backend.translator import get_translate
from backend.general_utils import download_ntlk

sys.path.pop(0)


class TextToSpeech:
    """
    Text to speech
    """
    @staticmethod
    def get_synthesized_audio(text, model_type, models, dir_time, **options):
        try:
            download_ntlk()  # inspect what ntlk downloaded

            results = TextToSpeech.get_models_results(text, model_type, models, dir_time, **options)
            return 0, results
        except Exception as err:
            print(f"Error when get synthesized audio... {err}")
            return 1, str(err)

    @staticmethod
    def get_models_results(text, model_type, models, dir_time, **options):
        if not os.path.exists(dir_time):
            os.makedirs(dir_time)

        current_models = {model_type: models[model_type]}

        results = []
        for model_name, model in current_models.items():
            start = time()
            audio = model.synthesize(text, **options)
            filename = model.save(audio, dir_time)
            with open(filename, "rb") as f:
                audio_bytes = f.read()

            end = time()

            sample_rate = model.sample_rate
            duration = len(audio) / sample_rate

            results.append(
                {
                    "voice": model_name,
                    "sample_rate": sample_rate,
                    "duration_s": round(duration, 3),
                    "synthesis_time": round(end - start, 3),
                    "filename": filename,
                    "response_audio": audio_bytes
                }
            )

        return results


class VoiceCloneTranslate:
    """
    Real time voice clone and translate
    """

    @staticmethod
    def get_synthesized_audio(audio_file, encoder, synthesizer, signature, vocoder, save_folder,
                              text, src_lang, need_translate, tts_model_name="Cloning voice", **options):
        try:
            download_ntlk()  # inspect what ntlk downloaded

            if need_translate:
                print("Translation text before voice clone")
                text = get_translate(text, src_lang)

            results = VoiceCloneTranslate.get_models_results(
                audio_file,
                text,
                encoder,
                synthesizer,
                signature,
                vocoder,
                save_folder,
                tts_model_name,
                **options
            )
            return 0, results
        except Exception as err:
            print(f"Error ... {err}")
            return 1, str(err)

    @staticmethod
    def get_models_results(audio_file, text, encoder, synthesizer, signature, vocoder, save_folder, tts_model_name, **options):
        from speech.rtvc_models import clone_voice_rtvc
        from speech.rtvc.speed.inference import AudioSpeedProcessor

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        start = time()

        # get voice for audio to use praat processing in wav format TODO device set?
        audio_file_voice = AudioSeparatorVoice.get_audio_separator(audio_file, save_folder, converted_wav=True, target="vocals", device="cpu")

        # clone voice
        clone_voice_rtvc(audio_file_voice, text, encoder, synthesizer, vocoder, save_folder)

        output_name = str(uuid.uuid4()) + ".wav"
        rtvc_output_file = VoiceCloneTranslate.merge_audio_parts(save_folder, "rtvc_output_part", output_name)
        output_file = AudioSpeedProcessor().process_and_save(audio_file_voice, rtvc_output_file)  # set speed from original

        end = time()

        with open(output_file, "rb") as f:
            audio_bytes = f.read()

        try:
            output_file_signature = signature.set_encrypted(output_file, save_folder)
            if output_file_signature is not None:
                os.remove(output_file)
                output_file = output_file_signature
        except Exception as err:
            print(f"Error...during set signature {err}")

        result = {
            "voice": tts_model_name,
            "sample_rate": 0,
            "duration_s": 0,
            "synthesis_time": round(end - start, 3),
            "filename": output_file,
            "response_audio": audio_bytes
        }

        return result

    @staticmethod
    def merge_audio_parts(audio_folder: str, audio_part_name: str, output_file_name: str):
        """
        Merge RTVC part files to one
        :param audio_folder: audio part folder and save folder
        :param audio_part_name: audio part name
        :param output_file_name: output audio merged file name
        :return: output audio merged file path
        """
        from speech.rtvc.encoder.audio import trim_silence_librosa

        # List all files in the directory
        files = os.listdir(audio_folder)

        # Filter out the relevant files and sort them
        relevant_files = sorted([f for f in files if f.startswith(audio_part_name) and f.endswith(".wav")])
        trim_relevant_files = []

        # File output path
        output_file_path = os.path.join(audio_folder, output_file_name)

        if not relevant_files:
            print("No matching files found during merge voice clone audio")
            return

        # Create a text file that lists all the .wav files to be concatenated
        merged_files = os.path.join(audio_folder, "merged_files.txt")
        with open(merged_files, "w") as f:
            for wav_file in relevant_files:
                trim_wav_path = trim_silence_librosa(os.path.join(audio_folder, wav_file), os.path.join(audio_folder, f"trimmed_{wav_file}"))
                trim_relevant_files.append(trim_wav_path)  # append already full path
                f.write(f"file '{trim_wav_path}'\n")

        # Use ffmpeg to concatenate the .wav files
        subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", os.path.join(audio_folder, "merged_files.txt"),
            "-c", "copy",
            output_file_path
        ])

        # Optionally, remove the temporary merged_files.txt file
        os.remove(merged_files)

        for wav_file in relevant_files:
            os.remove(os.path.join(audio_folder, wav_file))
        for trim_wav_file in trim_relevant_files:
            os.remove(trim_wav_file)

        print(f"Merged all .wav files into {output_file_name}")

        return output_file_path


class AudioSeparatorVoice:
    """
    Separate voice and noise from audio
    """
    @staticmethod
    def get_audio_separator(source, output_path, file_type="audio", converted_wav=True, target="vocals", device="cpu", trim_silence=True, resample=True):
        from speech.unmix.utils.model import AudioSeparator

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if file_type == "video":
            extracted_audio = AudioSeparatorVoice.extract_audio_from_video(source, output_path)
            if not extracted_audio:
                raise ValueError("Unable to extract audio from the provided video")
            source = os.path.join(output_path, extracted_audio)

        separator = AudioSeparator()
        output_file = separator.separate_audio(source, output_path, converted_wav=converted_wav, target_wav=target, device=device, resample=resample)
        # trim silence before analysis
        if trim_silence:
            source_output_file = output_file
            output_file = separator.trim_silence(source_output_file, output_path)
            os.remove(source_output_file)
        return output_file

    @staticmethod
    def extract_audio_from_video(video_path, save_path):
        # If not a GIF, proceed with audio extraction
        file_name = str(uuid.uuid4()) + '.wav'
        save_file = os.path.join(save_path, file_name)
        cmd = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{save_file}" -y'
        os.system(cmd)
        return file_name