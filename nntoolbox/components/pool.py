import torch
from torch import nn


class AveragePool(nn.Module):
    def __init__(self, dim):
        super(AveragePool, self).__init__()
        self._dim = dim

    def forward(self, input):
        return torch.mean(input, dim=self._dim)


class MaxPool(nn.Module):
    def __init__(self, dim):
        super(MaxPool, self).__init__()
        self._dim = dim

    def forward(self, input):
        return torch.max(input, dim=self._dim).values()


class ConcatPool(nn.Module):
    def __init__(self, pool_dim, concat_dim):
        super(ConcatPool, self).__init__()
        self._pool_dim = pool_dim
        self._concat_dim = concat_dim - 1 if pool_dim < concat_dim and concat_dim > 0 else concat_dim

    def forward(self, input):
        max = torch.max(input, dim=self._pool_dim).values
        avg = torch.mean(input, dim=self._pool_dim)
        return torch.cat([max, avg], dim=self._concat_dim)