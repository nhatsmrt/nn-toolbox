import torch
from torch import nn, Tensor
from typing import List


__all__ = ['Multiply', 'Mean', 'Sum']


class Multiply(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, input: Tensor) -> Tensor:
        return torch.stack([module(input) for module in self.module_list], dim=-1).prod(dim=-1)


class Sum(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, input: Tensor) -> Tensor:
        return torch.stack([module(input) for module in self.module_list], dim=-1).sum(dim=-1)


class Mean(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, input: Tensor) -> Tensor:
        return torch.stack([module(input) for module in self.module_list], dim=-1).mean(dim=-1)
