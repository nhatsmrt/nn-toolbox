import torch
from torch import nn
from torch.nn import functional as F
from ...components import HighwayLayer

class ConvolutionalLayer(nn.Sequential):
    '''
    Simple convolutional layer: input -> conv2d -> activation -> batch norm 2d
    '''
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, padding=0,
            bias=False, activation=nn.ReLU
    ):
        super(ConvolutionalLayer, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                ),
                activation(),
                nn.BatchNorm2d(num_features=out_channels)
            )
        )


class CoordConv2D(nn.Conv2d):
    '''
    Implement CoordConv
    https://arxiv.org/pdf/1807.03247.pdf
    '''
    def __init__(
            self, in_channels, out_channels,
            kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=True, padding_mode='zeros'
    ):
        super(CoordConv2D, self).__init__(
            in_channels + 2, out_channels,
            kernel_size, stride, padding,
            dilation, groups, bias, padding_mode
        )

    def forward(self, input):
        augmented_input = CoordConv2D.augment_input(input)
        return super().forward(augmented_input)

    @staticmethod
    def augment_input(input):
        '''
        Add two coordinate channels to input
        :param input: (N, C, H, W)
        :return: (N, C + 2, H, W)
        '''
        batch_size = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]

        i = torch.arange(0, w).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, h, 1).float()
        if w > 1:
            i = (i - (w - 1) / 2) / ((w - 1) / 2)

        j = torch.arange(0, h).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, w).float()
        if h > 1:
            j = (j - (h - 1) / 2) / ((h - 1) / 2)

        return torch.cat((input, i, j), dim=1)


class CoordConvolutionalLayer(nn.Sequential):
    '''
    Simple convolutional layer: input -> conv2d -> activation -> batch norm 2d
    '''
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, padding=0,
            bias=False, activation=nn.ReLU
    ):
        super(CoordConvolutionalLayer, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                CoordConv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                ),
                activation(),
                nn.BatchNorm2d(num_features=out_channels)
            )
        )

class HighwayConvolutionalLayer(HighwayLayer):
    '''
    Highway layer (for images):
    y = T(x) * H(x) + (1 - T(x)) * x
    '''
    def __init__(self, in_channels, main):
        '''
        :param in_channels: Number of channels of each input
        :param main: The main network H(x). Return output of same number of channels and dimensions
        '''
        super(HighwayConvolutionalLayer, self).__init__(
            in_features=in_channels,
            main=main,
            gate=ConvolutionalLayer(in_channels, in_channels, 3, padding=1, activation=nn.Sigmoid)
        )

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class ResizeConvolutionalLayer(nn.Module):
    '''
    Upsample the image (using an interpolation algorithm), then pass to a conv layer
    '''
    def __init__(self, in_channels, out_channels, mode='bilinear'):
        super(ResizeConvolutionalLayer, self).__init__()
        self._mode = mode
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, input, out_h, out_w):
        upsampled = F.interpolate(input, size=(out_h, out_w), mode=self._mode)
        return self._modules["conv"](upsampled)


class Reshape(nn.Module):
    def forward(self, input, new_shape):
        return input.view(new_shape)
