from torch import nn, Tensor
import torch
from ...utils import compute_jacobian
from typing import Tuple, Optional


__all__ = ['ShakeShake', 'DoubleBackpropagation']


class ShakeShake(nn.Module):
    """
    Implement shake-shake regularizer:

    y = x + sum_i alpha_i branch_i

    (alpha_i > 0 are random variables such that sum_i alpha_i = 1)

    At test time:

    y = x + 1 / n_branch sum_i branch_i

    Based on https://arxiv.org/abs/1705.07485
    """
    def __init__(self, keep: str='shake'):
        super(ShakeShake, self).__init__()
        self._keep = keep

    def forward(self, branches: Tensor, training: bool) -> Tensor:
        return ShakeShakeFunction.apply(branches, training, self._keep)


class ShakeShakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, branches: Tensor, training: bool, mode: str) -> Tensor:
        """
        :param ctx: context (to save info for backward pass)
        :param branches: outputs of all branches concatenated (cardinality, batch_size, n_channel, h, w)
        :param training: boolean, true if is training
        :param mode: 'keep': keep the forward weights for backward; 'even': backward with 1/n_branch weight;
                      'shake': randomly choose new weights
        :return: weighted sum of all branches' outputs
        """
        if training:
            branch_weights = ShakeShakeFunction.get_branch_weights(
                len(branches), branches[0].shape[0]
            ).to(branches.dtype).to(branches.device)
            output = branch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * branches

            if mode == 'keep':
                ctx.save_for_backward(torch.ones(1), branch_weights)
            elif training and mode == 'even':
                ctx.save_for_backward(-torch.ones(1), len(branches) * torch.ones(1).int())
            else:  # shake mode
                ctx.save_for_backward(torch.zeros(1), len(branches) * torch.ones(1).int())

            return torch.sum(output, dim=0)
        else:
            return torch.mean(branches, dim=0)

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if ctx.saved_tensors[0] == 1: # keep mode
            branch_weights = ctx.saved_tensors[1]
        elif ctx.saved_tensors[0] == -1: # even mode:
            cardinality = ctx.saved_tensors[1].item()
            branch_weights = 1.0 / cardinality * torch.ones(
                size=(cardinality, grad_output.shape[0])
            ).to(grad_output.device)
        else: # shake mode
            cardinality = ctx.saved_tensors[1].item()
            branch_weights = ShakeShakeFunction.get_branch_weights(
                cardinality, grad_output.shape[0]
            ).to(grad_output.device)

        return branch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(grad_output.dtype) * grad_output, None, None

    @staticmethod
    def get_branch_weights(cardinality: Tensor, batch_size: Tensor) -> Tensor:
        branch_weights = torch.rand(size=(cardinality, batch_size))
        branch_weights /= torch.sum(branch_weights, dim=0, keepdim=True)
        return branch_weights


class DoubleBackpropagation(nn.Module):
    """
    Double backpropagation regularizer to penalize slight perturbation in input

    https://www.researchgate.net/profile/Harris_Drucker/publication/5576575_Improving_generalization_performance_using_double_backpropagation/links/540754510cf2c48563b2ab7f.pdf

    http://yann.lecun.com/exdb/publis/pdf/drucker-lecun-91.pdf
    """
    def __init__(self, model: nn.Module, criterion: nn.Module):
        super(DoubleBackpropagation, self).__init__()
        self.main = model
        self.criterion = criterion

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        input.requires_grad = True
        jacobian = compute_jacobian(input, lambda input: self.criterion(self.model(input), label), True)
        input.requires_grad = False
        return 0.5 * torch.sum(jacobian * jacobian)
