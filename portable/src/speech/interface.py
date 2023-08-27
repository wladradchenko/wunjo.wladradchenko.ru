import os
import sys
import time
import logging
import subprocess

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from backend.translator import get_translate

sys.path.pop(0)


class TextToSpeech:
    """
    Text to speech
    """
    @staticmethod
    def get_synthesized_audio(text, model_type, models, dir_time, **options):
        try:
            results = TextToSpeech.get_models_results(text, model_type, models, dir_time, **options)
            return 0, results
        except Exception as e:
            logging.exception(e)
            return 1, str(e)

    @staticmethod
    def get_models_results(text, model_type, models, dir_time, **options):
        if not os.path.exists(dir_time):
            os.makedirs(dir_time)

        current_models = {model_type: models[model_type]}

        results = []
        for model_name, model in current_models.items():
            start = time.time()
            audio = model.synthesize(text, **options)
            filename = model.save(audio, dir_time)
            with open(filename, "rb") as f:
                audio_bytes = f.read()

            end = time.time()

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
    def split_into_phrases(text: str, max_words: int = 20, min_words: int = 10) -> list:
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

    @staticmethod
    def get_synthesized_audio(audio_file, encoder, synthesizer, vocoder, save_folder,
                              text, src_lang, need_translate, tts_model_name="Voice Clone", **options):
        try:
            if need_translate:
                print("Translation text before Voice Clone")
                text = get_translate(text, src_lang)

            results = VoiceCloneTranslate.get_models_results(
                audio_file,
                text,
                encoder,
                synthesizer,
                vocoder,
                save_folder,
                tts_model_name,
                **options
            )
            return 0, results
        except Exception as e:
            logging.exception(e)
            return 1, str(e)

    @staticmethod
    def get_models_results(audio_file, text, encoder, synthesizer, vocoder, save_folder, tts_model_name, **options):
        from speech.rtvc_models import clone_voice_rtvc

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        start = time.time()

        phrases = VoiceCloneTranslate.split_into_phrases(text)
        for i, phrase in enumerate(phrases):
            clone_voice_rtvc(audio_file, phrase, encoder, synthesizer, vocoder, save_folder, i)

        waves_format = ".wav"
        output_name = "rtvc_output" + VoiceCloneTranslate.uniqid() + time.strftime("%Y-%m-%d_%H-%M") + waves_format
        output_file = VoiceCloneTranslate.merge_audio_parts(save_folder, "rtvc_output_part", output_name)

        end = time.time()

        with open(output_file, "rb") as f:
            audio_bytes = f.read()

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
        # List all files in the directory
        files = os.listdir(audio_folder)

        # Filter out the relevant files and sort them
        relevant_files = sorted([f for f in files if f.startswith(audio_part_name) and f.endswith(".wav")])

        # File output path
        output_file_path = os.path.join(audio_folder, output_file_name)

        if not relevant_files:
            print("No relevant files found")
            return

        # Create a text file that lists all the .wav files to be concatenated
        merged_files = os.path.join(audio_folder, "merged_files.txt")
        with open(merged_files, "w") as f:
            for wav_file in relevant_files:
                f.write(f"file '{os.path.join(audio_folder, wav_file)}'\n")

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

        print(f"Merged all .wav files into {output_file_name}")

        return output_file_path

    @staticmethod
    def uniqid():
        from time import time
        return hex(int(time() * 1e7))[2:]
