import os
import sys
import uuid
import random

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.path.join(root_path, "tacotron2"))
from backend.folders import CUSTOM_VOICE_FOLDER
from tacotron2.train import main as tacotron2
from tacotron2.hparams import read_default_hparams, save_hparams
sys.path.pop(0)


class Tacotron2Train:
    """
    Train model for tacotron2
    """
    @staticmethod
    def train(param):
        train_path = Tacotron2Train.create_train_folder()
        checkpoint = Tacotron2Train.pre_train_checkpoint(param)
        mark = Tacotron2Train.get_mark(param)
        mark_test_file, mark_train_file = Tacotron2Train.markup_generation(mark, param, train_path)
        if mark_test_file and mark_train_file:
            config_path = Tacotron2Train.config_generation(param, mark_test_file, mark_train_file, train_path, checkpoint)
            if config_path is not None:
                Tacotron2Train.run_train(config_path)
                print(f"Train is finished! The model path is {train_path}")
                print(f"Visit the documentation at the https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki to know how to add the trained model to the application")
            else:
                Tacotron2TrainError("Error... Problem with creation config file to train. Training will not start!")
        else:
            raise Tacotron2TrainError("Error... Clear mark file is not include right format data to start train")

    @staticmethod
    def run_train(config_path):
        tacotron2(hparams_path=config_path)

    @staticmethod
    def config_generation(param, mark_test_file, mark_train_file, train_path, checkpoint):
        config = read_default_hparams()
        config["audios_path"] = param.get("audio_path")
        config["validation_files"] = mark_test_file
        config["training_files"] = mark_train_file
        config["output_dir"] = train_path
        config["charset"] = param.get("language", "en")
        config["batch_size"] = int(param.get("batch_size", 32))
        config["checkpoint"] = checkpoint
        config_path = save_hparams(train_path, config)
        return config_path

    @staticmethod
    def pre_train_checkpoint(param):
        checkpoint = param.get("checkpoint")
        if checkpoint:
            checkpoint = None if not os.path.isfile(checkpoint) else checkpoint
            return checkpoint
        return None

    @staticmethod
    def get_mark(param):
        mark_path = param.get("mark_path")
        if mark_path:
            mark_path = None if not os.path.isfile(mark_path) else mark_path
        if mark_path is None:
            raise Tacotron2TrainError("Error... Mark file is not found to training")
        with open(mark_path, "r", encoding="utf-8") as file:
            mark = file.readlines()
        return mark

    @staticmethod
    def markup_generation(mark, param, train_path):
        audio_path = param.get("audio_path")
        ratio = param.get("train_split")
        correct_mark = []
        for line in mark:
            # get what is right separate | in mark
            if "|" in line:
                split_line = line.split("|")
                if len(split_line) == 2:
                    audio_name, audio_text = split_line
                    audio_file = os.path.join(str(audio_path), audio_name)
                    audio_file = None if not os.path.isfile(audio_file) else audio_file
                    if audio_file is None:
                        print(f"Warning... Audio file is not exist and will not use in train {audio_file}")
                    else:
                        correct_mark.append(f"{audio_name}|{audio_text}")
                else:
                    print(f"Warning... Line has to be format: audio_file_name | text_of_audio. Not correct line will not use {line}")
            else:
                print(f"Warning... Delimiter has to be | in mark. Not correct line will not use {line}")

        if len(correct_mark) > 10:
            random.shuffle(correct_mark)  # random shuffle list
            mark_test_file, mark_train_file = Tacotron2Train.split_mark(correct_mark, ratio, train_path)
            return mark_test_file, mark_train_file
        return None, None

    @staticmethod
    def split_mark(mark, ratio, train_path):
        len_mark = len(mark)
        train_size = int(len_mark * (int(ratio) / 100))

        mark_test = mark[train_size:]
        mark_test_file = os.path.join(train_path, "mark_test_file.txt")
        with open(mark_test_file, "w", encoding="utf-8") as file:
            file.writelines(mark_test)

        mark_train = mark[:train_size]
        mark_train_file = os.path.join(train_path, "mark_train_file.txt")
        with open(mark_train_file, "w", encoding="utf-8") as file:
            file.writelines(mark_train)

        return mark_test_file, mark_train_file

    @staticmethod
    def create_train_folder() -> str:
        folder_name = str(uuid.uuid4())
        folder_path = os.path.join(CUSTOM_VOICE_FOLDER, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path


class Tacotron2TrainError(Exception):
    pass
