from torch.optim.lr_scheduler import ReduceLROnPlateau
from .callbacks import Callback
from torch.optim import Optimizer
from typing import Dict, Any


__all__ = ['LRSchedulerCB', 'ReduceLROnPlateauCB']


class LRSchedulerCB(Callback):
    def __init__(self, scheduler, timescale: str="iter"):
        assert timescale == "epoch" or timescale == "iter"
        self._scheduler = scheduler
        self._timescale = timescale

    def on_batch_end(self, logs: Dict[str, Any]):
        if self._timescale == "iter":
            self._scheduler.step()

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._timescale == "epoch":
            self._scheduler.step()
        return False


class ReduceLROnPlateauCB(Callback):
    def __init__(
            self, optimizer: Optimizer, monitor: str='accuracy',
            mode: str='max', factor: float=0.1, patience: int=10,
            verbose: bool=True, threshold: float=0.0001, threshold_mode: str='rel',
            cooldown: int=0, min_lr: float=0, eps: float=1e-08
    ):
        self._scheduler = ReduceLROnPlateau(optimizer, mode, factor, patience, verbose,
            threshold, threshold_mode, cooldown, min_lr, eps)
        self._monitor = monitor

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if "epoch_metrics" in logs:
            assert self._monitor in logs["epoch_metrics"]
            self._scheduler.step(logs["epoch_metrics"][self._monitor])
        return False
