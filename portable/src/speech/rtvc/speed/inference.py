import os
import subprocess
from parselmouth.praat import run_file


class AudioSpeedProcessor:
    high_lim_speed_factor = 1.5
    low_lim_speed_factor = 0.4

    def __init__(self):
        self.script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "speed_update.praat")
        print(self.script_path)

    def _audio_analysis(self, dir, file):
        sound = os.path.join(dir, file)
        try:
            objects = run_file(self.script_path, -20, 2, 0.27, "yes", sound, dir, 80, 400, 0.01, capture_output=True, return_variables=True)
            totDur = objects[2]['originaldur']
            nPause = objects[2]['npause']
            arDur = objects[2]['speakingtot']
            nSyl = objects[2]['voicedcount']
            arRate = objects[2]['articulationrate']
        except:
            totDur, nPause, arDur, nSyl, arRate = 0, 0, 0, 0, 0
            print("Try again; the sound of the audio was not clear.")
        return round(totDur, 2), int(nPause), round(arDur, 2), int(nSyl), round(arRate, 2)

    def process_and_save(self, original_audio, synthesized_audio):
        totDur_ori, nPause_ori, arDur_ori, nSyl_ori, arRate_ori = self._audio_analysis(*os.path.split(original_audio))
        path_syn, filename_syn = os.path.split(synthesized_audio)
        name_syn, suffix_syn = os.path.splitext(filename_syn)
        totDur_syn, nPause_syn, arDur_syn, nSyl_syn, arRate_syn = self._audio_analysis(path_syn, filename_syn)

        # Calculate speed factor
        if arRate_syn == 0:
            print("Exception! The speed factor is abnormal.")
            return synthesized_audio

        speed_factor = round(arRate_ori / arRate_syn, 2)

        if not (self.low_lim_speed_factor <= speed_factor <= self.high_lim_speed_factor):
            print("Exception! The speed factor is outside the acceptable range.")
            return synthesized_audio

        # Adjust speed using ffmpeg
        out_file = os.path.join(path_syn, f"{name_syn}_{speed_factor}{suffix_syn}")
        cmd = f"ffmpeg -i {synthesized_audio} -filter:a atempo={speed_factor} {out_file}"
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Finished! The path of out_file is {out_file}")
        return out_file