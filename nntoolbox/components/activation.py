import torch
from torch import nn, Tensor, cos
from ..utils import to_onehotv2


__all__ = ['ZeroCenterRelu', 'LWTA', 'Snake']


class ZeroCenterRelu(nn.ReLU):
    """
    As described by Jeremy of FastAI
    """
    def __init__(self, inplace: bool=False):
        super(ZeroCenterRelu, self).__init__(inplace)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input) - 0.5


class LWTA(nn.Module):
    """
    Local Winner-Take-All Layer

    For every k consecutive units, keep only the one with highest activations and zero-out the rest.

    References:
        Rupesh Kumar Srivastava, Jonathan Masci, Sohrob Kazerounian, Faustino Gomez, Jürgen Schmidhuber.
        "Compete to Compute." https://papers.nips.cc/paper/5059-compete-to-compute.pdf
    """
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape[1] % self.block_size == 0
        input = input.view(-1, input.shape[1] // self.block_size, self.block_size)
        mask = to_onehotv2(torch.max(input, -1)[1], self.block_size).to(input.dtype).to(input.device)
        return (input * mask).view(-1, input.shape[1] * input.shape[2])


class Snake(nn.Module):
    """
    Snake activation function for learning periodic function:

    snake_a(x) = x + 1 / freq * sin^2(freq * x) = x + (1 - cos(2 * freq * x)) / (2 * freq)

    References:

        Liu Ziyin, Tilman Hartwig, Masahito Ueda. "Neural Networks Fail to Learn Periodic Functions and How to Fix It."
        https://arxiv.org/pdf/2006.08195.pdf
    """
    def __init__(self, freq: float=1.0, freq_trainable: bool=False):
        super().__init__()
        assert freq > 0.0
        self.freq = nn.parameter.Parameter(torch.zeros((1)) + freq, requires_grad=freq_trainable)

    def forward(self, input: Tensor) -> Tensor:
        return input + (1.0 - cos(2 * self.freq * input)) / (2 * self.freq)
