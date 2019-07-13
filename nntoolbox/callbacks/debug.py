"""
Implement a debug callback. Adapt from fastai course2 v3 notebook 11 a
"""
from .callbacks import Callback
from typing import Callable, Dict, Any
from torch import Tensor


CALLBACK_STEPS = [
    'on_train_begin', 'on_epoch_begin', 'on_batch_begin',
    'after_outputs', 'after_losses', 'on_backward_begin',
    'after_backward', 'after_step', 'on_batch_end',
    'on_epoch_end', 'on_train_end'

]


# UNTESTED
class DebugCallback(Callback):
    def __init__(self, step_to_debug: str, func):
        assert step_to_debug in CALLBACK_STEPS
        self.step_to_debug =  step_to_debug
        self.func = func

    def on_train_begin(self):
        if self.step_to_debug == 'on_train_begin':
            self.func(self.learner)

    def on_epoch_begin(self):
        if self.step_to_debug == 'on_epoch_begin':
            self.func(self.learner)

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        if self.step_to_debug == 'on_batch_begin':
            self.func(self.learner)
        return data

    def after_outputs(self, outputs: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        if self.step_to_debug == 'after_outputs':
            self.func(self.learner)
        return outputs

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        if self.step_to_debug == 'after_losses':
            self.func(self.learner)
        return losses

    def on_backward_begin(self) -> bool:
        if self.step_to_debug == 'on_backward_begin':
            self.func(self.learner)
        return True # if false, skip backward

    def after_backward(self) -> bool:
        if self.step_to_debug == 'after_backward':
            self.func(self.learner)
        return True # whether to continue with iteration

    def after_step(self) -> bool:
        if self.step_to_debug == 'after_step':
            self.func(self.learner)
        return True

    def on_batch_end(self, logs: Dict[str, Any]):
        if self.step_to_debug == 'on_batch_end':
            self.func(self.learner)

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self.step_to_debug == 'on_epoch_end':
            self.func(self.learner)
        return super().on_epoch_end(logs) # whether to stop training

    def on_train_end(self):
        if self.step_to_debug == 'on_train_end':
            self.func(self.learner)
