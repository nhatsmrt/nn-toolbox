import torch
from torch import nn


class FactoredConvolutionalLayer(nn.Sequential):
    """
    Factor a k x k convolution into 1 x k and k x 1 inception style

    Help reduce memory footprint
    """
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
            activation=nn.ReLU, normalization=nn.BatchNorm2d
    ):
        if not isinstance(kernel_size, tuple): kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple): stride = (stride, stride)
        if not isinstance(padding, tuple): padding = (padding, padding)
        if not isinstance(dilation, tuple): dilation = (dilation, dilation)

        layers = [
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=(1, kernel_size[1]), stride=(1, stride[1]),
                padding=(0, padding[1]),
                dilation=(1, dilation[1]),
                groups=groups,
                bias=bias,
                padding_mode=padding_mode
            ),
            activation(),
            normalization(num_features=out_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(kernel_size[0], 1), stride=(stride[0], 1),
                padding=(padding[0], 0),
                dilation=(dilation[0], 1),
                groups=groups,
                bias=bias,
                padding_mode=padding_mode
            ),
            activation(),
            normalization(num_features=out_channels)
        ]
        super(FactoredConvolutionalLayer, self).__init__(*layers)

