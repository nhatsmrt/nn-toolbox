import torch
from torch import nn
from fastai.layers import AdaptiveConcatPool2d
from .layers import Flatten
from typing import Sequence

class FeedforwardBlock(nn.Sequential):
    def __init__(
            self, in_channels:int, hidden_layer_sizes:Sequence, out_features:int,
            pool_output_size:int, activation:nn.Module=nn.ReLU, drop_p=0.0
    ):
        layers = [AdaptiveConcatPool2d(sz=pool_output_size), Flatten()]
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                in_features = in_channels * 2 * pool_output_size * pool_output_size
            else:
                in_features = hidden_layer_sizes[i - 1]
            layers.append(nn.BatchNorm1d(num_features=in_features, momentum=0.01)) # follows fastai
            layers.append(nn.Dropout(p=drop_p))
            layers.append(nn.Linear(
                in_features=in_features,
                out_features=hidden_layer_sizes[i]
            ))
            layers.append(activation())

        layers.append(nn.Linear(in_features=hidden_layer_sizes[-1], out_features=out_features))
        super(FeedforwardBlock, self).__init__(*layers)