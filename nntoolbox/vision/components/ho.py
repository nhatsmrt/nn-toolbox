"""Some higher order layers"""
import torch
from torch import nn, Tensor


__all__ = ['QuadraticPolynomialConv2D']


class QuadraticPolynomialConv2D(nn.Module):
    """
    h(x) = sum_k(A_k * x)^2 + b * x + c

    where the * represents convolution

    References:

        Bergstra et al. "Quadratic Polynomials Learn Better Image Features."
        http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205 (dead link, use web archive)
    """
    def __init__(
            self, in_channels, out_channels, kernel_size, rank: int, stride=1, padding=0,
            dilation=1, groups=1, bias=True, padding_mode='zeros', sqrt: bool=False, eps: float=1e-6
    ):
        super().__init__()
        self.linear = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )
        self.quadratic = nn.Conv2d(
            in_channels, out_channels * rank, kernel_size, stride, padding, dilation, groups, False, padding_mode
        )
        self.out_channels = out_channels
        self.rank = rank
        self.sqrt = sqrt
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        linear_features = self.linear(input)
        quadratic_features = self.quadratic(input).pow(2)
        quadratic_features = quadratic_features.view(
            -1, self.rank, self.out_channels, quadratic_features.shape[2], quadratic_features.shape[3]
        ).sum(1)
        if self.sqrt:
            quadratic_features = torch.sqrt(quadratic_features + self.eps)
        return quadratic_features + linear_features
