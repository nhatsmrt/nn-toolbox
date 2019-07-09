import torch
from torch import nn
from fastai.layers import AdaptiveConcatPool2d
from .layers import Flatten
from typing import Sequence


class FeedforwardBlock(nn.Sequential):
    def __init__(
            self, in_channels:int, out_features:int, pool_output_size:int,
            hidden_layer_sizes:Sequence=(512,), activation:nn.Module=nn.ReLU,
            normalization=nn.BatchNorm1d, bn_final:bool=False, drop_p=0.5
    ):
        layers = [AdaptiveConcatPool2d(sz=pool_output_size), Flatten()]
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                in_features = in_channels * 2 * pool_output_size * pool_output_size
            else:
                in_features = hidden_layer_sizes[i - 1]
            layers.append(normalization(num_features=in_features))
            if drop_p != 0:
                layers.append(nn.Dropout(p=drop_p / 2))
            layers.append(nn.Linear(
                in_features=in_features,
                out_features=hidden_layer_sizes[i]
            ))
            layers.append(activation())
        if bn_final:
            layers.append(normalization(num_features=hidden_layer_sizes[-1], momentum=0.001)) #follows fast ai
        if drop_p != 0:
            layers.append(nn.Dropout(p=drop_p))
        layers.append(nn.Linear(in_features=hidden_layer_sizes[-1], out_features=out_features))
        super(FeedforwardBlock, self).__init__(*layers)
