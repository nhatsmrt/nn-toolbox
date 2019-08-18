"""Learning rate warmup (UNTESTED)"""
from .callbacks import Callback
from typing import Dict, Any


__all__ = ['LRWarmup', 'ConstantLRWarmup', 'GradualLRWarmup']


class LRWarmup(Callback):
    """
    Gradually increasing the LR at the beginning of training

    References:

        Priya Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour."
        https://arxiv.org/abs/1706.02677
    """
    def __init__(self, duration: int, timescale: str="iter"):
        self.order = 99
        self.duration = duration
        self.timescale = timescale
        self.cur = 0

    def on_batch_end(self, logs: Dict[str, Any]):
        if self.timescale == "iter":
            if self.cur < self.duration:
                self.update_lr()

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self.timescale == "epoch":
            if self.cur < self.duration:
                self.update_lr()
        return False

    def update_lr(self):
        for param_group in self.learner._optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        self.cur += 1

    def get_lr(self) -> float: pass


class ConstantLRWarmup(LRWarmup):
    def __init__(self, min_lr, duration: int, timescale: str="iter"):
        super().__init__(duration, timescale)
        self.min_lr = min_lr

    def get_lr(self) -> float: return self.min_lr


class GradualLRWarmup(LRWarmup):
    def __init__(self, min_lr: float, max_lr: float, duration: int, timescale: str="iter"):
        assert min_lr < max_lr
        super().__init__(duration, timescale)
        self.min_lr, self.max_lr = min_lr, max_lr

    def get_lr(self) -> float: return self.min_lr + (self.max_lr - self.min_lr) * (self.cur / self.duration)
