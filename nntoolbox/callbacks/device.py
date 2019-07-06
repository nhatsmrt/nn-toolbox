from .callbacks import Callback
from typing import Dict
from torch import Tensor
from ..utils import get_device


__all__ = ['ToDeviceCallback']


class ToDeviceCallback(Callback):
    def __init__(self, device=get_device()):
        self._device = device
        self.learner = None

    def on_train_begin(self):
        self.learner._model = self.learner._model.to(self._device)

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        for key in data:
            data[key] = data[key].to(self._device)
        return data
