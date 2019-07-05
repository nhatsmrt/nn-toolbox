from .callbacks import Callback
from typing import Dict, Any
from torch import Tensor
from ..utils import is_nan
from warnings import warn


__all__ = ['NaNWarner']


class NaNWarner(Callback):
    def on_batch_end(self, logs: Dict[str, Any]):
        for key in logs:
            if isinstance(logs[key], Tensor) and is_nan(logs[key]):
                warn(key + " becomes NaN at iteration " + str(logs["iter_cnt"]))


class TerminateOnNaN(Callback):
    def on_batch_end(self, logs: Dict[str, Any]):
        for key in logs:
            if isinstance(logs[key], Tensor) and is_nan(logs[key]):
                raise ValueError(key + " becomes NaN at iteration " + str(logs["iter_cnt"]))
