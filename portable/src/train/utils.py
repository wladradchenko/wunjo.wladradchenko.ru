import os
import sys
import time
import torch

from train import Tacotron2Train


def clear_memory():
    torch.cuda.empty_cache()
    time.sleep(10)
    torch.cuda.empty_cache()


def training_route(param):
    clear_memory()  # clear cache for torch cuda
    type_train = param.get("train_type")
    if type_train == "tacotron2":
        Tacotron2Train.train(param)
