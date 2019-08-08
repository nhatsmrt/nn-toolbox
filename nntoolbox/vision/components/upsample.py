import torch
from torch import nn
from torch.nn import functional as F
from .layers import ConvolutionalLayer


__all__ = ['ResizeConvolutionalLayer', 'PixelShuffleConvolutionLayer']


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

    References:

        https://arxiv.org/pdf/1609.05158.pdf

        https://arxiv.org/pdf/1806.02658.pdf (additional blurring at the end)
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
        :param: whether to blur at the end to remove checkerboard artifact
        """
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * (upscale_factor ** 2),
            # kernel_size=3,
            kernel_size=1,
            padding=0,
        )
        self.initialize_conv(conv, in_channels, out_channels, upscale_factor)
        layers = [
            # nn.ReplicationPad2d(1),
            conv,
            activation(),
            normalization(num_features=out_channels * (upscale_factor ** 2)),
            nn.PixelShuffle(upscale_factor)
        ]
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(kernel_size=2, stride=1)]
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
        weight_tensor = torch.rand(out_channels, in_channels, 1, 1)
        kaiming_uniform_(weight_tensor, a=math.sqrt(5))
        weight_tensor = weight_tensor.repeat((upscale_factor ** 2, 1, 1, 1))
        conv.weight.data.copy_(weight_tensor)
