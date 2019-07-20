from .callbacks import Callback
from typing import Dict, Any
from torch import Tensor
from ..utils import is_nan
from warnings import warn


__all__ = ['NaNWarner', 'SkipNaN']


class NaNWarner(Callback):
    def on_batch_end(self, logs: Dict[str, Any]):
        for key in logs:
            if isinstance(logs[key], Tensor) and is_nan(logs[key]):
                warn(key + " becomes NaN at iteration " + str(logs["iter_cnt"]))


class SkipNaN(Callback):
    """
    Skip when loss or output is nan (UNTESTED)
    """
    def after_outputs(self, outputs: Dict[str, Tensor], train: bool) -> bool:
        for key in outputs:
            if is_nan(outputs[key]):
                print("One of the loss is nan. Skip")
                return False

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> bool:
        for key in losses:
            if is_nan(losses[key]):
                print("One of the losses is nan. Skip")
                self.learner._optimizer.zero_grad()
                return False


class TerminateOnNaN(Callback):
    """
    Terminate training when encounter NaN (INCOMPLETE)
    """
    def on_batch_end(self, logs: Dict[str, Any]):
        for key in logs:
            if isinstance(logs[key], Tensor) and is_nan(logs[key]):
                raise ValueError(key + " becomes NaN at iteration " + str(logs["iter_cnt"]))
