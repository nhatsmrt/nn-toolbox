from torch import nn
from nntoolbox.vision.components.layers import ConvolutionalLayer
from nntoolbox.vision.components.regularization import ShakeShake
from .kervolution import KervolutionalLayer
import torch


class ResNeXtBlock(nn.Module):
    """
    Implement a resnext block:

    y = x + sum_i branch_i
    """
    def __init__(self, branches, use_shake_shake):
        super(ResNeXtBlock, self).__init__()
        self._use_shake_shake = use_shake_shake
        self.branches = branches
        self._cardinality = len(self.branches)

        if use_shake_shake:
            self._shake_shake = ShakeShake()

    def forward(self, input):
        branches_outputs = torch.stack([self.branches[i](input) for i in range(self._cardinality)], dim=0)
        if self._use_shake_shake:
            return input + self._shake_shake(branches_outputs, self.training)
        else:
            return input + torch.sum(branches_outputs, dim=0)


class _ResidualBlockNoBN(nn.Module):
    """
    Residual Block without the final Batch Normalization layer
    """
    def __init__(self, in_channels):
        super(_ResidualBlockNoBN, self).__init__()
        self._main = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self._main(input) + input


class ResidualBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                _ResidualBlockNoBN(in_channels),
                nn.BatchNorm2d(in_channels)
            )
        )


class ResidualBlockPreActivation(ResNeXtBlock):
    def __init__(self, in_channels, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(ResidualBlockPreActivation, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=1,
                            activation=activation, normalization=normalization
                        )
                    )
                ]
            ),
            use_shake_shake=False
        )


class BottleneckPreActivation(ResNeXtBlock):
    def __init__(self, in_channels, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(BottleneckPreActivation, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        ConvolutionalLayer(
                            in_channels, in_channels // 4, 1, padding=0,
                            activation=activation, normalization=normalization
                        ),
                        ConvolutionalLayer(
                            in_channels // 4, in_channels // 4, 3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        ConvolutionalLayer(
                            in_channels // 4, in_channels, 1, padding=0,
                            activation=activation, normalization=normalization
                        )
                    )
                ]
            ),
            use_shake_shake=False
        )


class ResidualBlockPreActivationKer(ResNeXtBlock):
    def __init__(self, in_channels, kernel, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(ResidualBlockPreActivationKer, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        KervolutionalLayer(
                            in_channels, in_channels, kernel=kernel, kernel_size=3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        KervolutionalLayer(
                            in_channels, in_channels, kernel=kernel, kernel_size=3, padding=1,
                            activation=activation, normalization=normalization
                        )
                    )
                ]
            ),
            use_shake_shake=False
        )



