import torch
from torch import nn
from .pool import GlobalAveragePool
from .res import _ResidualBlockNoBN

class SEBlock(nn.Module):
    '''
    Implement squeeze (global information embedding) and excitation (adaptive recalibration) mechanism
    https://arxiv.org/pdf/1709.01507.pdf
    '''
    def __init__(self, in_channels, reduction_ratio):
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
    def __init__(self, in_channels, reduction_ratio):
        super(_SEResidualBlockNoBN, self).__init__(in_channels)
        self._se = SEBlock(in_channels, reduction_ratio)

    def forward(self, input):
        return self._se(self._main(input)) + input

class SEResidualBlock(nn.Sequential):
    def __init__(self, in_channels, reduction_ratio):
        super(SEResidualBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                _SEResidualBlockNoBN(in_channels, reduction_ratio),
                nn.BatchNorm2d(in_channels)
            )
        )
