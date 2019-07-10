from .callbacks import Callback
from typing import Dict, List, Union, Optional
from torch import Tensor, device
from torch.nn import DataParallel
from ..utils import get_device
from torchgpipe import GPipe
from torchgpipe_balancing import balance_by_time


__all__ = ['ToDeviceCallback', 'ToGPipeDeviceCallback']


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


class ToGPipeDeviceCallback(Callback):
    """
    Set up for training using multiple gpus with GPipe algorithm
    Using kakaobrain's torchgpipe library:
    https://torchgpipe.readthedocs.io/en/stable/
    Reference:
    https://arxiv.org/pdf/1811.06965.pdf
    """
    def __init__(self, input_keys, target_keys, partitions: int, sample: Tensor, chunks: int=16):
        self.learner = None
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.partitions = partitions
        self.sample = sample
        self.chunks = chunks

    def on_train_begin(self):
        balance = balance_by_time(self.learner._model, self.sample, partitions=self.partitions)
        self.learner._model = GPipe(self.learner._model, balance=balance, chunks=self.chunks)

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        in_device = self.learner._model.devices[0]
        out_device = self.learner._model.devices[-1]
        for key in data:
            if key in self.input_keys:
                data[key] = data[key].to(in_device, non_blocking=True)
            elif key in self.target_keys:
                data[key] = data[key].to(out_device, non_blocking=True)
        return data


# UNTESTED
class DataParallelismCallback(Callback):
    """
    Callback for naive data parallelism: copy model to each device, then divide each batch into micro-batch for
    independent processing
    """
    def __init__(
            self, device_ids: Optional[List[Union[int, device]]]=None,
            output_device: Optional[Union[int, device]]=None, dim: int=0
    ):
        self.learner = None
        self.device_ids = device_ids
        self.ouput_device = output_device
        self.dim = dim

    def on_train_begin(self):
        self._learner.model = DataParallel(self._learner.model, self.device_ids, self.ouput_device, self.dim)
