from torch.nn import MSELoss
import torch.nn.functional as F
from torch import Tensor, nn
import torch
from typing import List, Optional


__all__ = ['RMSELoss', 'LogSigmoidLoss', 'CombinedLoss']


class RMSELoss(MSELoss):
    def __init__(self, reduction='mean', eps: float=1e-8):
        super(RMSELoss, self).__init__(reduction=reduction)
        self._eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(super().forward(input, target) + self._eps)


class LogSigmoidLoss(nn.Module):
    def forward(self, input: Tensor) -> Tensor: return -F.logsigmoid(input).mean(0)


class CombinedLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]]=None):
        super(CombinedLoss, self).__init__()
        if weights is not None: assert len(weights) == len(losses)
        else: weights = [1.0 / len(losses) for _ in range(len(losses))]
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        losses = torch.stack([self.losses[i](input, target) * self.weights[i] for i in range(len(self.losses))], -1)
        return torch.sum(losses)
