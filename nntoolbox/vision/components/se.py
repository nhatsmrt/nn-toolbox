from ...utils import copy_model
from torch import nn
from .layers import ConvolutionalLayer
from .pool import GlobalAveragePool
from .res import _ResidualBlockNoBN, ResNeXtBlock
from .kervolution import KervolutionalLayer


class SEBlock(nn.Module):
    '''
    Implement squeeze (global information embedding) and excitation (adaptive recalibration) mechanism
    https://arxiv.org/pdf/1709.01507.pdf
    '''
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        # self._res = ResidualBlock()
        self._squeeze_excitation = nn.Sequential(
            GlobalAveragePool(),
            nn.Linear(in_features=in_channels, out_features=in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_features=in_channels // reduction_ratio, out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, input):
        channel_weights = self._squeeze_excitation(input)
        return channel_weights.unsqueeze(-1).unsqueeze(-1) * input


class _SEResidualBlockNoBN(_ResidualBlockNoBN):
    def __init__(self, in_channels, reduction_ratio=16):
        super(_SEResidualBlockNoBN, self).__init__(in_channels)
        self._se = SEBlock(in_channels, reduction_ratio)

    def forward(self, input):
        return self._se(self._main(input)) + input


class SEResidualBlock(nn.Sequential):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEResidualBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                _SEResidualBlockNoBN(in_channels, reduction_ratio),
                nn.BatchNorm2d(in_channels)
            )
        )


class SEResidualBlockPreActivation(ResNeXtBlock):
    def __init__(self, in_channels, reduction_ratio=16, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(SEResidualBlockPreActivation, self).__init__(
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
                        ),
                        SEBlock(in_channels, reduction_ratio)
                    )
                ]
            ),
            use_shake_shake=False
        )


class SEResidualBlockPreActivationKer(ResNeXtBlock):
    def __init__(self, in_channels, kernel, reduction_ratio=16, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(SEResidualBlockPreActivationKer, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        KervolutionalLayer(
                            in_channels, in_channels, kernel, 3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        KervolutionalLayer(
                            in_channels, in_channels, kernel, 3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        SEBlock(in_channels, reduction_ratio)
                    )
                ]
            ),
            use_shake_shake=False
        )
