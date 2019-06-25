from torch.nn import Module, MSELoss
from torch import Tensor
import torch


class RMSELoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', eps: float=1e-8):
        super(RMSELoss, self).__init__()
        self.base_loss = MSELoss(size_average, reduce, reduction)
        self._eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.base_loss(input, target) + self._eps)
