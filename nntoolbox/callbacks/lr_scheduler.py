from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from .callbacks import Callback
from torch.optim import Optimizer
from typing import Dict, Any


class LRSchedulerCB(Callback):
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        self._scheduler.step()
        return False


class ReduceLROnPlateauCB(Callback):
    def __init__(
            self, optimizer: Optimizer, monitor: str='accuracy', mode: str='max', factor=0.1, patience=10,
            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    ):
        self._scheduler = ReduceLROnPlateau(optimizer, mode, factor, patience, verbose,
            threshold, threshold_mode, cooldown, min_lr, eps)
        self._monitor = monitor

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if "epoch_metrics" in logs:
            assert self._monitor in logs["epoch_metrics"]
            self._scheduler.step(logs["epoch_metrics"][self._monitor])
        return False