"""Scaling the learning rate layerwise (HIGHLY EXPERIMENTAL)"""


from torch.optim import SGD, Adam
import torch
from typing import Callable, Tuple
from torch import Tensor


__all__ = ['LARS', 'LAMB']


class LARS(SGD):
    """
    Implement Layer-wise Adaptive Rate Scaling (LARS) algorithm for training with large batch and learning rate

    References:

        https://arxiv.org/pdf/1708.03888.pdf
    """

    def __init__(
            self, params, lr: float,  momentum: float=0.0, weight_decay: float=0.0,
            trust_coefficient: float=0.001, eps: float=1e-8
    ):
        super(LARS, self).__init__(
            params, lr, momentum, 0.0, weight_decay, False
        )
        self.trust_coefficient, self.eps = trust_coefficient, eps

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                data_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)
                local_lr = (
                    self.trust_coefficient * data_norm / (grad_norm + weight_decay * data_norm + self.eps)
                ).detach()

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = (group['lr'] * local_lr * torch.clone(d_p)).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(group['lr'] * local_lr, d_p)

                    d_p = buf

                p.data.add_(-d_p)

        return loss


class LAMB(Adam):
    """
    Implement LAMB algorithm for training with large batch and learning rate

    Note that in second version of the paper, bias correction for betas is missing.

    References:

        https://arxiv.org/pdf/1904.00962.pdf
    """
    def __init__(
            self, params, lr: float=1e-3, betas: Tuple[float, float]=(0.9, 0.999),
            eps: float=1e-8, weight_decay: float=0,
            scaling_fn: Callable[[Tensor], Tensor] = lambda x: x,
            amsgrad: bool=False, correct_bias: bool=True
    ):
        super(LAMB, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.scaling_fn = scaling_fn
        self.correct_bias = correct_bias

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                weight_decay_term = group['weight_decay'] * p.data  # decouple the weight decay

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if self.correct_bias:
                    exp_avg = exp_avg / (1 - beta1 ** state['step'])
                    exp_avg_sq = exp_avg_sq / (1 - beta2 ** state['step'])

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                direction = exp_avg / denom + weight_decay_term
                # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                step_size = group['lr'] * self.scaling_fn(p.data.norm()) / (direction.norm() + group['eps'])

                p.data.add_(-step_size, direction)

        return loss


