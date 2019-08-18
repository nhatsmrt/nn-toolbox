"""A few regularizers, implemented as callbacks (UNTESTED)"""
import torch
from torch import Tensor
from .callbacks import Callback
from ..hooks import OutputHook
from torch.autograd import grad
from typing import Dict, Callable


__all__ = [
    'WeightRegularization', 'WeightElimination', 'L1WR', 'L2WR',
    'ActivationRegularization', 'L1AR', 'L2AR',
    'StudentTPenaltyAR', 'DoubleBackpropagationCB'
]


class WeightRegularization(Callback):
    """Regularization by penalizing weights"""
    def __init__(self, regularizer: Callable[[Tensor], Tensor], lambd: float, loss_name: str='loss'):
        self.loss_name = loss_name
        self.regularizer = regularizer
        self.lambd = lambd

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        assert self.loss_name in losses
        reg = 0.0
        for p in self.learner._model.parameters():
            reg = reg + self.regularizer(p.data)
        losses[self.loss_name] += self.lambd * reg
        return losses


class WeightElimination(WeightRegularization):
    def __init__(self, scale: float, lambd: float, loss_name: str='loss'):
        assert scale > 0.0

        def weight_elimination(t: Tensor) -> Tensor:
            t_sq = t.pow(2)
            return t_sq / (t_sq + scale ** 2).sum()

        super().__init__(
            regularizer=weight_elimination,
            lambd=lambd, loss_name=loss_name
        )


class L1WR(WeightRegularization):
    def __init__(self, lambd: float, loss_name: str='loss'):
        super(L1WR, self).__init__(
            regularizer=lambda t: t.float().abs().sum(),
            lambd=lambd,
            loss_name=loss_name
        )


class L2WR(WeightRegularization):
    def __init__(self, lambd: float, loss_name: str='loss'):
        super(L2WR, self).__init__(
            regularizer=lambda t: t.float().pow(2).sum(),
            lambd=lambd,
            loss_name=loss_name
        )


class ActivationRegularization(Callback):
    """Regularization by penalizing activations"""
    def __init__(
            self, output_hook: OutputHook,
            regularizer: Callable[[Tensor], Tensor],
            lambd: float, loss_name: str='loss'
    ):
        """
        :param output_hook: output hook of the module we want to regularize
        :param regularizer: regularization function (e.g L2)
        :param loss_name: name of the loss stored in loss logs. Default to 'loss'
        """
        self.hook = output_hook
        self.loss_name = loss_name
        self.regularizer = regularizer
        self.lambd = lambd

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        assert self.loss_name in losses
        losses[self.loss_name] += self.regularizer(self.hook.store) * self.lambd
        self.hook.store = None
        return losses


class L2AR(ActivationRegularization):
    def __init__(self, output_hook: OutputHook, lambd: float, loss_name: str='loss'):
        super(L2AR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.float().pow(2).sum(),
            lambd=lambd,
            loss_name=loss_name
        )


class L1AR(ActivationRegularization):
    def __init__(self, output_hook: OutputHook, lambd: float, loss_name: str='loss'):
        super(L1AR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.float().abs().sum(),
            lambd=lambd,
            loss_name=loss_name
        )


class StudentTPenaltyAR(ActivationRegularization):
    """
    Student's T Activation Regularization:

    omega(t) = sum_i log(1 + t_i^2)
    """
    def __init__(self, output_hook: OutputHook, lambd: float, loss_name: str='loss'):
        super(StudentTPenaltyAR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: torch.log1p(t.pow(2)).mean(),
            lambd=lambd,
            loss_name=loss_name
        )


class DoubleBackpropagationCB(Callback):
    """
    Double backpropagation regularizer to penalize slight perturbation in input (as a CB) (UNTESTED)

    https://www.researchgate.net/profile/Harris_Drucker/publication/5576575_Improving_generalization_performance_using_double_backpropagation/links/540754510cf2c48563b2ab7f.pdf

    http://yann.lecun.com/exdb/publis/pdf/drucker-lecun-91.pdf

    http://luizgh.github.io/libraries/2018/06/22/pytorch-doublebackprop/
    """
    def __init__(self, input_name: str="inputs", loss_name: str="loss"):
        self.input_name = input_name
        self.loss_name = loss_name

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        data[self.input_name].requires_grad = True
        self.input = data[self.input_name]
        return data

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        inp_grad = grad(losses[self.loss_name], self.input, create_graph=True)[0]
        self.input.requires_grad = False
        self.input = None
        losses[self.loss_name] = losses[self.loss_name] + 0.5 * inp_grad.pow(2).sum()
        return losses
