"""A few regularizers, implemented as callbacks (UNTESTED)"""
import torch
from torch import Tensor
from .callbacks import Callback
from ..hooks import OutputHook
from typing import Dict, Callable


class ActivationRegularization(Callback):
    def __init__(self, output_hook: OutputHook, regularizer: Callable[[Tensor], Tensor], loss_name: str='loss'):
        """
        :param output_hook: output hook of the module we want to regularize
        :param regularizer: regularization function (e.g L2)
        :param loss_name: name of the loss stored in loss logs. Default to 'loss'
        """
        self.hook = output_hook
        self.loss_name = loss_name
        self.regularizer = regularizer

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        assert self.loss_name in losses
        losses[self.loss_name] += self.regularizer(self.hook.store)
        return losses


class L2AR(ActivationRegularization):
    def __init__(self, output_hook: OutputHook, loss_name: str='loss'):
        super(L2AR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.float().pow(2).mean(),
            loss_name=loss_name
        )


class L1AR(ActivationRegularization):
    def __init__(self, output_hook: OutputHook, loss_name: str='loss'):
        super(L1AR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.float().abs().mean(),
            loss_name=loss_name
        )


class StudentTPenaltyAR(ActivationRegularization):
    """
    Student's T Activation Regularization:

    omega(t) = sum_i log(1 + t_i^2)
    """
    def __init__(self, output_hook: OutputHook, loss_name: str='loss'):
        super(StudentTPenaltyAR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: torch.log1p(t.pow(2)).mean(),
            loss_name=loss_name
        )
