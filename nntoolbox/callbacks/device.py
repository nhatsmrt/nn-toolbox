from .callbacks import Callback
from typing import Dict, List, Union, Optional, Callable
from torch import Tensor, device
from torch.nn import DataParallel, Module, AdaptiveAvgPool2d, Sequential
from ..utils import get_device, cut_model


__all__ = ['ToDeviceCallback']


class ToDeviceCallback(Callback):
    def __init__(self, device=get_device()):
        self._device = device
        self.learner = None

    def on_train_begin(self):
        if hasattr(self.learner, '_model'):
            self.learner._model = self.learner._model.to(self._device)
        elif hasattr(self.learner, '_models'):
            for i in range(len(self.learner._models)):
                self.learner._models[i] = self.learner._models[i].to(self._device)

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        for key in data:
            data[key] = data[key].to(self._device)
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
        self.device_ids = device_ids
        self.ouput_device = output_device
        self.dim = dim

    def on_train_begin(self):
        self.learner._model = DataParallel(self.learner._model, self.device_ids, self.ouput_device, self.dim)


class MixedParallelismCB(Callback):
    """
    Callback for mixed parallelism: data parallelism for convolution/feature layers, and model parallelism for head
    (UNTESTED)

    References:

    https://discuss.pytorch.org/t/why-not-giving-the-whole-model-to-dataparallel-in-the-imagenet-example/4092

    https://arxiv.org/pdf/1404.5997.pdf
    """
    def __init__(
            self, device_ids: Optional[List[Union[int, device]]]=None,
            output_device: Optional[Union[int, device]]=None, dim: int=0,
            sep: Callable[..., Module]=AdaptiveAvgPool2d
    ):
        self.device_ids = device_ids
        self.ouput_device = output_device
        self.dim = dim
        self.sep = sep

    def on_train_begin(self):
        features, head = cut_model(self.learner._model, sep=self.sep)
        features = DataParallel(features, self.device_ids, self.ouput_device, self.dim)
        self.learner._model = Sequential(features, head).to(self.ouput_device)

