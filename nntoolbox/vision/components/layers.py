import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ...components import HighwayLayer
from typing import Callable


__all__ = [
    'LambdaLayer', 'ConvolutionalLayer', 'CoordConv2D', 'CoordConvolutionalLayer',
    'HighwayConvolutionalLayer', 'Flatten', 'ResizeConvolutionalLayer', 'PixelShuffleConvolutionLayer',
    'Reshape', 'InputNormalization'
]


class LambdaLayer(nn.Module):
    """
    Implement a quick layer wrapper for a function
    Useful for stateless layer (e.g without parameters)
    """
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        super(LambdaLayer, self).__init__()
        self.fn = fn

    def forward(self, input):
        return self.fn(input)


class ConvolutionalLayer(nn.Sequential):
    """
    Simple convolutional layer: input -> conv2d -> activation -> norm 2d
    """
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, padding=0,
            bias=False, activation=nn.ReLU, normalization=nn.BatchNorm2d
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
                normalization(num_features=out_channels)
            )
        )


class CoordConv2D(nn.Conv2d):
    """
    Implement CoordConv
    https://arxiv.org/pdf/1807.03247.pdf
    """
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
        """
        Add two coordinate channels to input
        :param input: (N, C, H, W)
        :return: (N, C + 2, H, W)
        """
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
    """
    Simple convolutional layer: input -> conv2d -> activation -> batch norm 2d
    """
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
    """
    Highway layer (for images):
    y = T(x) * H(x) + (1 - T(x)) * x
    """
    def __init__(self, in_channels, main):
        """
        :param in_channels: Number of channels of each input
        :param main: The main network H(x). Return output of same number of channels and dimensions
        """
        super(HighwayConvolutionalLayer, self).__init__(
            in_features=in_channels,
            main=main,
            gate=ConvolutionalLayer(in_channels, in_channels, 3, padding=1, activation=nn.Sigmoid)
        )


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResizeConvolutionalLayer(nn.Module):
    """
    Upsample the image (using an interpolation algorithm), then pass to a conv layer
    """
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, normalization=nn.BatchNorm2d, mode='bilinear'):
        super(ResizeConvolutionalLayer, self).__init__()
        self._mode = mode
        self.conv = ConvolutionalLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
            normalization=normalization
        )

    def forward(self, input, out_h, out_w):
        upsampled = F.interpolate(input, size=(out_h, out_w), mode=self._mode)
        return self.conv(upsampled)


class PixelShuffleConvolutionLayer(nn.Sequential):
    """
    Upsample the image using normal convolution follow by pixel shuffling
    https://arxiv.org/pdf/1609.05158.pdf
    """
    def __init__(
            self, in_channels: int, out_channels: int, upscale_factor: int, activation=nn.ReLU,
            normalization=nn.BatchNorm2d, blur: bool=True
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param upscale_factor: factor to increase spatial resolution by
        :param activation: activation function
        :param normalization: normalization function
        """
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=0,
        )
        self.initialize_conv(conv, in_channels, out_channels, upscale_factor)
        layers = [
            nn.ReplicationPad2d(1),
            conv,
            activation(),
            normalization(num_features=out_channels * (upscale_factor ** 2)),
            nn.PixelShuffle(upscale_factor)
        ]
        if blur:
            layers += [nn.ReplicationPad2d(1), nn.AvgPool2d(kernel_size=2, stride=1)]
        super(PixelShuffleConvolutionLayer, self).__init__(*layers)

    def initialize_conv(self, conv, in_channels: int, out_channels: int, upscale_factor: int):
        """
        Initialize according to:
        https://arxiv.org/pdf/1707.02937.pdf
        :param conv:
        :param in_channels:
        :param out_channels:
        :param upscale_factor:
        :return:
        """
        from torch.nn.init import kaiming_uniform_
        import math
        weight_tensor = torch.rand(out_channels, in_channels, 3, 3)
        kaiming_uniform_(weight_tensor, a=math.sqrt(5))
        weight_tensor = weight_tensor.repeat((upscale_factor ** 2, 1, 1, 1))
        conv.weight.data.copy_(weight_tensor)


class Reshape(nn.Module):
    def forward(self, input, new_shape):
        return input.view(new_shape)


class InputNormalization(nn.Module):
    """
    Normalize input before feed into a network
    Adapt from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """

    def __init__(self, mean, std):
        super(InputNormalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self._mean = torch.tensor(mean).view(-1, 1, 1)
        self._std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self._mean) / self._std

    def to(self, *args, **kwargs):
        self._mean = self._mean.to(*args, **kwargs)
        self._std = self._std.to(*args, **kwargs)
        super().to(*args, **kwargs)
