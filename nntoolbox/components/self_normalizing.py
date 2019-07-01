import torch
from torch import nn, Tensor
from typing import Sequence
from torch.nn.init import uniform_, _calculate_fan_in_and_fan_out
import math
import numpy as np


__all__ = ['SelfNormalizingMLP']


class SelfNormalizingMLP(nn.Sequential):
    '''
    Implement Self-Normalizing Neural Networks:
    https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf
    '''
    def __init__(
            self, in_features: int, out_features: int,
            hidden_layer_sizes: Sequence[int]=(512,), drop_ps=(0.5, 0.5)
    ):
        layers = []
        if isinstance(drop_ps, float):
            drop_ps = [drop_ps for _ in range(len(hidden_layer_sizes) + 1)]

        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                in_features = in_features
            else:
                in_features = hidden_layer_sizes[i - 1]
            drop_p = drop_ps[i]
            if drop_p != 0:
                layers.append(nn.AlphaDropout(p=drop_p))
            layers.append(SelfNormalizingLinear(
                in_features=in_features,
                out_features=hidden_layer_sizes[i],
                bias=True,
            ))
            layers.append(nn.SELU())

        if drop_ps[-1] != 0:
            layers.append(nn.AlphaDropout(p=drop_ps[-1]))
        layers.append(SelfNormalizingLinear(in_features=hidden_layer_sizes[-1], out_features=out_features, bias=True))
        super(SelfNormalizingMLP, self).__init__(*layers)


class SelfNormalizingLinear(nn.Linear):
    '''
    Linear layer with special initialization
    '''
    def reset_parameters(self):
        self_normalize_normal_(self.weight)
        if self.bias is not None:
            # fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in)
            uniform_(self.bias, -0.0, 0.0)


def self_normalize_normal_(tensor: Tensor):
    std = 1 / np.sqrt(tensor.shape[1])
    with torch.no_grad():
        return tensor.normal_(0.0, std)
