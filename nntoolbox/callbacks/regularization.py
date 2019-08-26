"""A few regularizers, implemented as callbacks (UNTESTED)"""
import torch
from torch import Tensor
from .callbacks import Callback
from ..hooks import OutputHook
from ..utils import compute_jacobian_v2, count_trainable_parameters, hessian_diagonal
from torch.autograd import grad
from typing import Dict, Callable


__all__ = [
    'WeightRegularization', 'WeightElimination', 'L1WR', 'L2WR',
    'ActivationRegularization', 'L1AR', 'L2AR', 'StudentTPenaltyAR',
    'TemporalActivationRegularization', 'L1TAR', 'L2TAR',
    'DoubleBackpropagationCB', 'FlatMinimaSearch'
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
        assert self.loss_name in losses
        losses[self.loss_name] += self.regularizer(self.hook.store) * self.lambd
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
            output, states = self.hook.store
            if isinstance(states, tuple): states = states[0]
            states_change = states[:len(states) - 1] - states[1:]
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


class FlatMinimaSearch(Callback):
    """
    Encourage model to find flat minima, which helps generalization (UNTESTED)

    References:
        Sepp Hochreiter and Jurgen Schmidhuber. "Flat Minima."
        https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.1.1

        Sepp Hochreiter and Jurgen Schmidhuber. "Feature Extraction through LOCOCODE."
        http://www.bioinf.jku.at/publications/older/2104.pdf
    """
    def __init__(self, lambd: float, output_name: str, loss_name: str, eps: float=1e-6):
        assert lambd >= 0.0
        self.lambd = lambd
        self.output_name = output_name
        self.loss_name = loss_name
        self.eps = eps

    def on_train_begin(self):
        self.n_params = count_trainable_parameters(self.learner._model)

    def after_outputs(self, outputs: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        assert self.output_name in outputs
        self.output = outputs[self.output_name]
        return outputs

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        assert self.loss_name in losses
        # batch_size = self.output.shape[0]
        total_loss = []

        for ind in range(len(self.output)):
            output = self.output[ind].view(-1)
            jacobian = compute_jacobian_v2(output, self.learner._model.parameters(), True)

            sq = [g.pow(2).sum(0) for g in jacobian]
            abs_jacob = [g.abs() for g in jacobian]
            sq_sqrt = [(g + self.eps).sqrt().unsqueeze(0) for g in sq]

            first_term = torch.stack([torch.log(g).sum() for g in sq], dim=0).sum()
            second_term = [abs_g / sq_sqrt_g for abs_g, sq_sqrt_g in zip(abs_jacob, sq_sqrt)]
            second_term = sum([torch.log(g.sum(0).pow(2)) for g in second_term]) * self.n_params
            total_loss.append(first_term + second_term)

        total_loss = torch.stack(total_loss, dim=0).mean()
        losses[self.loss_name] += total_loss * self.lambd
        self.output = None

        return losses


class CurvatureDrivenSmoothing(Callback):
    """
    Encourage the model to be smooth (i.e has small curvature) (UNTESTED)

    References:

        Chris M. Bishop. "Curvature-Driven Smoothing: A Learning Algorithm for Feedforward Networks."
        https://pdfs.semanticscholar.org/3c39/ddfbd9822230b5375d581bf505ecf6255283.pdf
    """
    def __init__(self, lambd: float, input_name: str, output_name: str, loss_name: str, eps: float=1e-6):
        assert lambd >= 0.0
        self.lambd = lambd
        self.output_name = output_name
        self.input_name = input_name
        self.loss_name = loss_name
        self.eps = eps

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        if train:
            assert self.input_name in data
            self.input = data[self.input_name]
        return data

    def after_outputs(self, outputs: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        if train:
            assert self.output_name in outputs
            self.output = outputs[self.output_name]
        return outputs

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        if train:
            assert self.loss_name in losses
            total_loss = []

            for ind in range(len(self.output)):
                output = self.output[ind].view(-1)
                hessian_diag = hessian_diagonal(output, self.input, True)
                sos = torch.stack([h.pow(2).sum() for h in hessian_diag], dim=0).sum(0)
                total_loss.append(sos)

            self.output = None
            self.input = None
            total_loss = torch.stack(total_loss, dim=0).mean() * 0.5
            losses[self.loss_name] += total_loss * self.lambd

        return losses

