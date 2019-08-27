from torch import nn
from ...components import GeneralizedShuntingModule
from .layers import BiasLayer2D


__all__ = ['SiConv2D']


class SiConv2D(GeneralizedShuntingModule):
    """
    Implement a shunting inhibition convolution layer. Right now only support channelwise fully connected variant.

        Difference from original implementation: clamping denominator.

    References:

        Fok Hing Chi Tivive and Abdesselam Bouzerdoum.
        "Efficient Training Algorithms for a Class of Shunting Inhibitory Convolutional Neural Networks."
        https://ieeexplore.ieee.org/document/1427760
    """
    def __init__(
            self, in_channels, out_channels, kernel_size,
            stride: int=1, padding=0, dilation=1,
            groups=1, bias=True, padding_mode='zeros',
            num_activation: nn.Module = nn.Identity(),
            denom_activation: nn.Module = nn.ReLU(),
            bound_denom: bool = True, bound: float = 0.1
    ):
        num = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode),
            num_activation
        )
        denom = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode),
            denom_activation,
            BiasLayer2D(out_channels, init=1.0)
        )
        super().__init__(num, denom, bound_denom, bound)
