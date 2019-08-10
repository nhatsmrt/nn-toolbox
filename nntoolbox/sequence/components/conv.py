"""Convolution modules for text classification"""
import torch
from torch import nn, Tensor
from typing import Union, List


__all__ = ['ConvolutionalLayer1D']


class ConvolutionalLayer1D(nn.Module):
    """
    Implement a layer that aggregates multiple conv1d of different filter sizes, then pooled over time.

    References:

        Yoon Kim. "Convolutional Neural Networks for Sentence Classification."
        https://aclweb.org/anthology/D14-1181

    """
    def __init__(
            self, in_channels: int, out_channels: Union[int, List[int]], kernel_sizes: List[int],
            strides: Union[int, List[int]]=1, paddings: Union[int, List[int]]=0, dilations: Union[int, List[int]]=1,
            groups: Union[int, List[int]]=1, biases: Union[bool, List[bool]]=True,
            padding_modes: Union[str, List[str]]='zeros', batch_first: bool=False
    ):
        super(ConvolutionalLayer1D, self).__init__()
        if isinstance(out_channels, list): assert len(out_channels) == len(kernel_sizes)
        else: out_channels = [out_channels for _ in range(len(kernel_sizes))]

        if isinstance(strides, list): assert len(strides) == len(kernel_sizes)
        else: strides = [strides for _ in range(len(kernel_sizes))]

        if isinstance(paddings, list): assert len(paddings) == len(kernel_sizes)
        else: paddings = [paddings for _ in range(len(kernel_sizes))]

        if isinstance(dilations, list): assert len(dilations) == len(kernel_sizes)
        else: dilations = [dilations for _ in range(len(kernel_sizes))]

        if isinstance(groups, list): assert len(groups) == len(kernel_sizes)
        else: groups = [groups for _ in range(len(kernel_sizes))]

        if isinstance(biases, list): assert len(biases) == len(kernel_sizes)
        else: biases = [biases for _ in range(len(kernel_sizes))]

        if isinstance(padding_modes, list): assert len(padding_modes) == len(kernel_sizes)
        else: padding_modes = [padding_modes for _ in range(len(kernel_sizes))]

        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels, out_channels[i], kernel_sizes[i], strides[i],
            paddings[i], dilations[i], groups[i], biases[i], padding_modes[i]
        ) for i in range(len(kernel_sizes))])
        self.batch_first = batch_first

    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 1) if self.batch_first else input.permute(1, 2, 0)
        return torch.cat([conv(input).max(-1)[0] for conv in self.convs], -1)
