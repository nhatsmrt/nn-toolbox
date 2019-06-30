from torch.nn import Module, MSELoss
from torch import Tensor
from nntoolbox.utils import to_onehot
import torch
import torch.nn.functional as F
from typing import Optional


__all__ = ['RMSELoss']


class RMSELoss(MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', eps: float=1e-8):
        super(RMSELoss, self).__init__(size_average, reduce, reduction)
        self._eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(super().forward(input, target) + self._eps)


