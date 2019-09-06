from .callbacks import Callback
from ..optim import change_lr, get_lr
import numpy as np
from typing import Dict, Any


__all__ = ['VariableLengthBPTT']


class VariableLengthBPTT(Callback):
    """
    Change the truncated backprop through time length and linearly scale the learning rate. (UNTESTED)

    References:

        Stephen Merity, Nitish Shirish Keskar, Richard Socher. "Regularizing and Optimizing LSTM Language Models."
        https://arxiv.org/abs/1708.02182
    """

    def __init__(self, default_len: int, p: float, std: float):
        assert 0.0 < p < 1.0
        assert std > 0.0
        self.default_len, self.p, self.std = default_len, p, std
        self.original_lr = None

    def on_epoch_begin(self):
        base_length = np.random.choice([self.default_len, self.default_len / 2], p=[self.p, 1.0 - self.p])
        epoch_length = min(max(int(np.random.normal(base_length)), 1), self.default_len)
        self.learner._train_iterator.bptt_len = epoch_length
        self.original_lr = get_lr(self.learner._optimizer)
        new_lr = [lr * epoch_length / self.default_len for lr in self.original_lr]
        change_lr(self.learner._optimizer, new_lr)

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        change_lr(self.learner._optimizer, self.original_lr)
        return False

