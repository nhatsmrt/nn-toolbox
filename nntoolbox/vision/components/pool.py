import torch
from torch import nn


__all__ = [
    'spatial_pyramid_pool', 'SpatialPyramidPool', 'GlobalAveragePool',
    'AdaptiveConcatPool2d'
]


def spatial_pyramid_pool(input, op_sizes, pool_layer=nn.MaxPool2d):
    """
    :param input: (batch_size, C, H, W)
    :param op_sizes:
    :param pool_layer:
    :return:
    """
    ops = []
    batch_size = input.shape[0]
    inp_h = input.shape[2]
    inp_w = input.shape[3]


    for size in op_sizes:
        pool = pool_layer(
            kernel_size=(torch.ceil(torch.Tensor([inp_h / size])), torch.ceil(torch.Tensor([inp_w / size]))),
            stride=(torch.floor(torch.Tensor([inp_h / size])), torch.floor(torch.Tensor([inp_w / size])))
        )
        ops.append(pool(input).view(batch_size, -1))

    # for op in ops:
    #     print(op.shape)
    return torch.cat(ops, dim = -1)


class SpatialPyramidPool(nn.Module):
    def __init__(self, op_sizes, pool_layer = nn.MaxPool2d):
        super(SpatialPyramidPool, self).__init__()
        self._op_sizes = op_sizes
        self._pool_layer = pool_layer

    def forward(self, input):
        return spatial_pyramid_pool(input, self._op_sizes, self._pool_layer)


class GlobalAveragePool(nn.Module):
    def __init__(self):
        super(GlobalAveragePool, self).__init__()
    def forward(self, input):
        return torch.mean(torch.mean(input, dim = -1), dim = -1)


class AdaptiveConcatPool2d(nn.Module):
    """
    Adapt from FastAI's code
    """

    def __init__(self, output_size):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=output_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.max_pool(input),
            self.avg_pool(input)
        ], dim=1)

