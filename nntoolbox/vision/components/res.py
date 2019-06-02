from torch import nn
from .layers import ConvolutionalLayer

class _ResidualBlockNoBN(nn.Sequential):
    '''
    Residual Block without the final Batch Normalization layer
    '''

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
