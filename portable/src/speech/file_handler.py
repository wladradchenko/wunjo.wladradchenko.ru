import os
import time
import logging


class FileHandler:
    @staticmethod
    def get_synthesized_audio(text, model_type, models, dir_time, **options):
        try:
            results = FileHandler.get_models_results(text, model_type, models, dir_time, **options)

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