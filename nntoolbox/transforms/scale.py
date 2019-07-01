from sklearn.preprocessing import StandardScaler
from ..utils import get_device
import torch
from torch import Tensor


__all__ = ['StandardScalerTransform']


class StandardScalerTransform:
    def __init__(self, scaler: StandardScaler=None, device=get_device()):
        self.scaler = StandardScaler() if scaler is None else scaler
        self._device = device

    def fit(self, inputs):
        self.scaler.fit(inputs)

    def __call__(self, input: Tensor) -> Tensor:
        return torch.from_numpy(self.scaler.transform(input[None, :])).to(input.dtype)[0]
