from torch.optim import SGD
import torch


__all__ = ['LARS']


class LARS(SGD):
    """
    Implement Layer-wise Adaptive Rate Scaling (LARS) algorithm for training with large batch and learning rate
    (UNTESTED)

    References:

        https://arxiv.org/pdf/1708.03888.pdf
    """

    def __init__(self, params, lr: float,  momentum: float=0.0, weight_decay: float=0.0, trust_coefficient: float=0.02):
        super(LARS, self).__init__(
            params, lr, momentum, 0.0, weight_decay, False
        )
        self.trust_coefficient = trust_coefficient

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
                local_lr = (self.trust_coefficient * data_norm / (grad_norm + weight_decay * data_norm)).detach()

                # print(grad_norm)
                # print(data_norm)
                # print(local_lr)
                # print()

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = (group['lr'] * local_lr * torch.clone(d_p)).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.mul_(momentum).add_(group['lr'] * local_lr, d_p)

                    d_p = buf

                p.data.add_(-d_p)

        return loss
