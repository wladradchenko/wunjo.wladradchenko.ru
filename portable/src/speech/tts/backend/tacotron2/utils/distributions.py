import math
from enum import Enum

import torch
from torch.nn.init import _calculate_correct_fan


class DistTypes(str, Enum):
    kaiming_uniform = "kaiming_uniform"
    xavier_uniform = "xavier_uniform"


def calculate_gain(nonlinearity, param=None):
    # https://github.com/pytorch/pytorch/issues/24991
    if nonlinearity == "selu":
        return 1.000707983970642
    else:
        return torch.nn.init.calculate_gain(nonlinearity, param)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def init_weights(tensor, dist_type, nonlinearity, **kwargs):
    if dist_type == DistTypes.kaiming_uniform:
        kaiming_uniform_(tensor, nonlinearity=nonlinearity, **kwargs)
    elif dist_type == DistTypes.xavier_uniform:
        torch.nn.init.xavier_uniform_(tensor, gain=calculate_gain(nonlinearity))
    else:
        raise TypeError