"""A few regularizers, implemented as callbacks (UNTESTED)"""
import torch
from torch import Tensor
from .callbacks import Callback
from ..hooks import OutputHook
from ..utils import compute_jacobian
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
        self.hook.store = None
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


class DoubleBackpropagationCB(Callback):
    """
    Double backpropagation regularizer to penalize slight perturbation in input (as a CB) (UNTESTED)

    https://www.researchgate.net/profile/Harris_Drucker/publication/5576575_Improving_generalization_performance_using_double_backpropagation/links/540754510cf2c48563b2ab7f.pdf

    http://yann.lecun.com/exdb/publis/pdf/drucker-lecun-91.pdf
    """
    def __init__(self, input_name: str='inputs', loss_name: str='loss'):
        self.loss_name = loss_name
        self.input_name = input_name

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        assert self.input_name in data
        self.input = data[self.input_name]
        return data

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        assert self.loss_name in losses
        self.input.requires_grad = True
        jacobian = compute_jacobian(self.input, lambda input: losses[self.loss_name], True)
        self.input.requires_grad = False
        self.input = None
        losses[self.loss_name] = losses[self.loss_name] + 0.5 * torch.sum(jacobian * jacobian)
        return losses

