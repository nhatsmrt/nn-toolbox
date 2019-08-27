"""Convolution and Quasi-RNN modules"""
import torch
from torch import nn, Tensor
from typing import Union, List, Tuple, Optional


__all__ = ['ConvolutionalLayer1D', 'MaskedConv1D', 'QRNNLayer']


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


class MaskedConv1D(nn.Conv1d):
    """
    Output at t should only depend on input from t - k + 1 to t

    References:

        James Bradbury, Stephen Merity, Caiming Xiong, Richard Socher. "Quasi-Recurrent Neural Networks."
        https://arxiv.org/abs/1611.01576
    """
    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (seq_len, batch_size, in_channels)
        :return: (seq_len, batch_size, out_channels)
        """
        mask = torch.zeros((self.kernel_size[0] - 1, input.shape[1], input.shape[2])).to(input.device).to(input.dtype)
        return super().forward(torch.cat([mask, input], dim=0).permute(1, 2, 0)).permute(2, 0, 1)


class QRNNLayer(nn.Module):
    """
    Quasi RNN layer. Decouple the gate computation (which can now be performed parallelwise with convolution)
    and the hidden state sequential computation.

    References:

        James Bradbury, Stephen Merity, Caiming Xiong, Richard Socher. "Quasi-Recurrent Neural Networks."
        https://arxiv.org/abs/1611.01576
    """
    def __init__(
            self, input_size: int, hidden_size: int, kernel_size: int, pooling_mode: str='fo'
    ):
        super().__init__()
        assert pooling_mode == 'f' or pooling_mode == 'fo' or pooling_mode == 'ifo'
        self.n_gates = len(pooling_mode) + 1
        self.pooling_mode = pooling_mode
        self.hidden_size = hidden_size
        out_channels = hidden_size * (len(pooling_mode) + 1)
        self.conv = MaskedConv1D(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, input: Tensor, h: Optional[Tensor]=None) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        if h is None: h = torch.zeros((1, input.shape[1], self.hidden_size)).to(input.device).to(input.dtype)
        h = h[0]
        gates = self.conv(input).chunk(self.n_gates, -1)

        if self.pooling_mode == 'f':
            z, f = torch.tanh(gates[0]), torch.sigmoid(gates[1])
        elif self.pooling_mode == 'fo':
            z, f, o = torch.tanh(gates[0]), torch.sigmoid(gates[1]), torch.sigmoid(gates[2])
        else:
            z, f, o, i = torch.tanh(gates[0]), torch.sigmoid(gates[1]), torch.sigmoid(gates[2]), torch.sigmoid(gates[3])

        hs = []
        if self.pooling_mode != 'f': c = torch.zeros(h.shape).to(input.device).to(input.dtype)

        for t in range(len(input)):
            if self.pooling_mode == 'f':
                h = f[t] * h + (1 - f[t]) * z[t]
            elif self.pooling_mode == 'fo':
                c = f[t] * c + (1 - f[t]) * z[t]
                h = c * o[t]
            else:
                c = f[t] * c + i[t] * z[t]
                h = c * o[t]

            hs.append(h)

        if self.pooling_mode == 'f':
            return torch.stack(hs, dim=0), h[None: ]
        else:
            return torch.stack(hs, dim=0), (h[None, :], c[None, :])
