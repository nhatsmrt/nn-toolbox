import torch
from torch import nn
from .layers import ConvolutionalLayer

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, activation):
        super(DenseLayer, self).__init__()

        self.add_module(
            "main",
            nn.Sequential(
                nn.BatchNorm2d(num_features = in_channels),
                activation(inplace = True),
                ConvolutionalLayer(
                    in_channels = in_channels,
                    out_channels = growth_rate,
                    kernel_size = 1,
                    stride = 1,
                    bias=False,
                    activation=activation
                ),
                nn.Conv2d(
                    in_channels = growth_rate,
                    out_channels = growth_rate,
                    kernel_size = 3,
                    stride = 1,
                    padding=1,
                    bias=False
                )
            )
        )

    def forward(self, input):
        return torch.cat((input, super(DenseLayer, self).forward(input)), dim = 1)

class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, num_layers, activation=nn.ReLU):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module(
                "DenseLayer_" + str(i),
                DenseLayer(
                    in_channels = in_channels + growth_rate * i,
                    growth_rate = growth_rate,
                    activation=activation
                )
            )
