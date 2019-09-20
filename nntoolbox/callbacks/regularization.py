"""A few regularizers, implemented as callbacks (UNTESTED)"""
import torch
from torch import Tensor
from .callbacks import Callback
from ..hooks import OutputHook
from typing import Dict, Callable


__all__ = [
    'WeightRegularization', 'WeightElimination', 'L1WR', 'L2WR',
    'ActivationRegularization', 'L1AR', 'L2AR', 'StudentTPenaltyAR',
    'TemporalActivationRegularization', 'L1TAR', 'L2TAR'
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
            regularizer=lambda t: t.norm(1).mean(),
            lambd=lambd,
            loss_name=loss_name
        )


class L2WR(WeightRegularization):
    def __init__(self, lambd: float, loss_name: str='loss'):
        super(L2WR, self).__init__(
            regularizer=lambda t: t.norm(2).mean(),
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
        if train:
            assert self.loss_name in losses
            outputs = self.hook.store
            if isinstance(outputs, tuple): outputs = outputs[0]
            losses[self.loss_name] += self.regularizer(outputs) * self.lambd
            self.hook.store = None
        return losses

    def on_train_end(self):
        self.hook.remove()


class L2AR(ActivationRegularization):
    def __init__(self, output_hook: OutputHook, lambd: float, loss_name: str='loss'):
        super(L2AR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.norm(2).mean(),
            lambd=lambd,
            loss_name=loss_name
        )


class L1AR(ActivationRegularization):
    def __init__(self, output_hook: OutputHook, lambd: float, loss_name: str='loss'):
        super(L1AR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.norm(1).mean(),
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


class LowActivityPrior(ActivationRegularization):
    """
    Constraint the activation to be small. Coupling with a variance force, this will drive the activation to sparsity.

    (UNTESTED)

    References:

        Sven Behnke. "Hierarchical Neural Networks for Image Interpretation," page 124.
        https://www.ais.uni-bonn.de/books/LNCS2766.pdf
    """
    def __init__(self, output_hook: OutputHook, lambd: float, alpha: float=0.1, loss_name: str='loss'):
        super(LowActivityPrior, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: (t.mean(0) - alpha).pow(2).mean(),
            lambd=lambd,
            loss_name=loss_name
        )


class TemporalActivationRegularization(Callback):
    """
    Regularizing by penalizing activation change.

    References:

        Stephen Merity, Bryan McCann, Richard Socher. "Revisiting Activation Regularization for Language RNNs."
        https://arxiv.org/pdf/1708.01009.pdf
    """
    def __init__(
            self, output_hook: OutputHook,
            regularizer: Callable[[Tensor], Tensor],
            lambd: float, loss_name: str = 'loss'
    ):
        self.lambd, self.loss_name, self.regularizer, self.hook = lambd, loss_name, regularizer, output_hook

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        if train:
            assert self.loss_name in losses
            outputs = self.hook.store
            if isinstance(outputs, tuple): outputs = outputs[0]
            states_change = outputs[:len(outputs) - 1] - outputs[1:]
            losses[self.loss_name] = self.regularizer(states_change) * self.lambd + losses[self.loss_name]
            self.hook.store = None

        return losses

    def on_train_end(self):
        self.hook.remove()


class L2TAR(TemporalActivationRegularization):
    def __init__(
            self, output_hook: OutputHook,
            lambd: float, loss_name: str = 'loss'
    ):
        super(L2TAR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.norm(2).mean(),
            lambd=lambd,
            loss_name=loss_name
        )


class L1TAR(TemporalActivationRegularization):
    def __init__(
            self, output_hook: OutputHook,
            lambd: float, loss_name: str = 'loss'
    ):
        super(L1TAR, self).__init__(
            output_hook=output_hook,
            regularizer=lambda t: t.norm(1).mean(),
            lambd=lambd,
            loss_name=loss_name
        )
