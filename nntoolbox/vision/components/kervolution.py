import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
Implement kervolution (kernel convolution) layers
https://arxiv.org/pdf/1904.03955.pdf
"""


class LinearKernel(nn.Module):
    def __init__(self, cp: float=1.0, trainable=True):
        assert cp > 0
        super(LinearKernel, self).__init__()
        self.log_cp = nn.Parameter(torch.tensor(np.log(cp), requires_grad=trainable))

    def forward(self, input: Tensor, weight: Tensor, bias: Tensor):
        weight = weight.view(weight.shape[0], -1).t()
        output = input.permute(0, 2, 1).matmul(weight).permute(0, 2, 1) + torch.exp(self.log_cp)

        return output + bias.unsqueeze(0).unsqueeze(-1) if bias is not None else output


class PolynomialKernel(LinearKernel):
    def __init__(self, dp: int=3, cp: float=2.0, trainable=True):
        super(PolynomialKernel, self).__init__(cp, trainable)
        self._dp = dp

    def forward(self, input: Tensor, weight: Tensor, bias: Tensor):
        return super().forward(input, weight, bias).pow(self._dp)


class GaussianKernel(nn.Module):
    def __init__(self, bandwidth: int=1, trainable=True):
        assert bandwidth > 0
        super(GaussianKernel, self).__init__()
        self.log_bandwidth = nn.Parameter(torch.tensor(np.log(bandwidth)), requires_grad=trainable)

    def forward(self, input: Tensor, weight: Tensor, bias: Tensor):
        """
        :param input: (batch_size, patch_size, n_patches)
        :param weight: (out_channels, in_channels, kernel_height, kernel_width)
        :return:
        """
        input = input.unsqueeze(-2)
        weight = weight.view(weight.shape[0], -1).t().unsqueeze(0).unsqueeze(-1)
        output = torch.exp(-torch.exp(self.log_bandwidth) * (input - weight).pow(2).sum(1))
        return output + bias.unsqueeze(0).unsqueeze(-1) if bias is not None else output


class Kervolution2D(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel, kernel_size, stride=1,
            padding=0, dilation=1,
            bias=True, padding_mode='zeros'
    ):
        super(Kervolution2D, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, 1,
            bias, padding_mode
        )
        self.kernel = kernel()

    def compute_output_shape(self, height, width):
        def compute_shape_helper(inp_dim, padding, kernel_size, dilation, stride):
            return np.floor(
                (inp_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            ).astype(np.uint32)
        return (
            compute_shape_helper(height, self.padding[0], self.kernel_size[0], self.dilation[0], self.stride[0]),
            compute_shape_helper(width, self.padding[1], self.kernel_size[1], self.dilation[1], self.stride[1]),
        )

    def forward(self, input):
        output_h, output_w = self.compute_output_shape(input.shape[2], input.shape[3])

        if self.padding_mode == 'circular':
            expanded_padding = [(self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2]
            input = F.pad(input, expanded_padding, mode='circular')
            padding = 0
        else:
            padding = self.padding
        input = F.unfold(
            input, kernel_size=self.kernel_size, dilation=self.dilation,
            padding=padding, stride=self.stride
        )
        output = self.kernel(input, self.weight, self.bias)
        # output = torch.clamp(output, min=-10.0, max=10.0)
        return output.view(-1, self.out_channels, output_h, output_w)


class KervolutionalLayer(nn.Sequential):
    """
    Simple convolutional layer: input -> conv2d -> activation -> norm 2d
    """
    def __init__(
            self, in_channels, out_channels, kernel,
            kernel_size=3, stride=1, padding=0,
            bias=False, activation=nn.ReLU, normalization=nn.BatchNorm2d
    ):
        super(KervolutionalLayer, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                Kervolution2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                ),
                activation(),
                normalization(num_features=out_channels)
            )
        )
