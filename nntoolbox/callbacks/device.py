from .callbacks import Callback
from typing import Dict
from torch import Tensor
from ..utils import get_device


__all__ = ['ToDeviceCallback', 'DataToGPipeDeviceCallback']


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


class DataToGPipeDeviceCallback(Callback):
    def __init__(self, input_keys, target_keys):
        self.learner = None
        self.input_keys = input_keys
        self.target_keys = target_keys

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        in_device = self.learner._model.devices[0]
        out_device = self.learner._model.devices[-1]
        for key in data:
            if key in self.input_keys:
                data[key] = data[key].to(in_device, non_blocking=True)
            elif key in self.target_keys:
                data[key] = data[key].to(out_device, non_blocking=True)
        return data
