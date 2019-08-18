"""Learning rate warmup (UNTESTED)"""
from .callbacks import Callback
from typing import Dict, Any
from torch import Tensor


__all__ = ['LRWarmup', 'ConstantLRWarmup', 'GradualLRWarmup']


class LRWarmup(Callback):
    """
    Start training with a small learning rate

    References:

        Priya Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour."
        https://arxiv.org/abs/1706.02677
    """
    def __init__(self, duration: int, timescale: str="iter"):
        self.order = 99
        self.duration = duration
        self.timescale = timescale
        self.cur = 0

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        if self.timescale == "iter":
            if self.cur < self.duration:
                self.update_lr()
        return data

    def on_epoch_begin(self):
        if self.timescale == "epoch":
            if self.cur < self.duration:
                self.update_lr()

    def update_lr(self):
        for param_group in self.learner._optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        self.cur += 1

    def get_lr(self) -> float: pass


class ConstantLRWarmup(LRWarmup):
    """Keeping the learning rate at a small value for several iterations/epochs"""
    def __init__(self, min_lr, duration: int, timescale: str="iter"):
        super().__init__(duration, timescale)
        self.min_lr = min_lr

    def get_lr(self) -> float: return self.min_lr


class GradualLRWarmup(LRWarmup):
    """Gradually increase the learning rate from a small value for several iterations/epochs"""
    def __init__(self, min_lr: float, max_lr: float, duration: int, timescale: str="iter"):
        assert min_lr < max_lr
        super().__init__(duration, timescale)
        self.min_lr, self.max_lr = min_lr, max_lr

    def get_lr(self) -> float: return self.min_lr + (self.max_lr - self.min_lr) * (self.cur / self.duration)
