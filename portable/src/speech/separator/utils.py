import os
import uuid
import librosa
import soundfile as sf


def trim_silence(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Trim the silence from the start and end
    # `top_db` is the threshold in dB below which audio is considered silent
    y_trimmed, index = librosa.effects.trim(y, top_db=60)

    # Save the trimmed audio
    output_file = os.path.join(output_path, str(uuid.uuid4()) + ".wav")
    sf.write(output_file, y_trimmed, sr)
    return output_file