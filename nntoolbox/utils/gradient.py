import torch
import numpy as np
from torch.nn import Module
from torch import Tensor
from torch.autograd import grad
from typing import List, Callable, Union, Iterable


__all__ = [
    'compute_gradient', 'compute_jacobian', 'compute_jacobian_v2',
    'update_gradient', 'accumulate_gradient', 'compute_gradient_norm',
    'hessian_diagonal', 'gather_flat_grad'
]


def compute_gradient(output: Tensor, model: Module) -> List[Tensor]:
    """
    Comput gradient of the output of a model

    :param output:
    :param model:
    :return: list of gradients of model parameters
    """
    ret = []
    output.backward(retain_graph=True)
    for parameter in model.parameters():
        ret.append(parameter.grad)
        parameter.grad = None # Reset gradient accumulation
    return ret


def compute_jacobian(
        input: Tensor, fn: Callable[[Tensor], Tensor], is_batch: bool=True, requires_grad: bool=True
) -> Tensor:
    """
    Compute the jacobian of function(input) with respect to input. For most purpose, should use v2

    :param output:
    :param input: assume that input require_grad = True
    :param fn:
    :param batch: whether to compute gradient by batch
    :return:
    """
    if is_batch:
        return torch.stack([compute_jacobian(input[ind], fn, False) for ind in range(len(input))], dim=0)
    else:
        output = fn(input)
        output_shape = output.shape
        input_shape = input.shape
        output = output.view(-1)
        grad = [
            torch.autograd.grad(output[ind], [input], allow_unused=True, create_graph=requires_grad)[0]
            for ind in range(len(output))
        ]
        return torch.stack(grad, dim=0).reshape(output_shape + input_shape)


def compute_jacobian_v2(
        output: Tensor, input: Union[Tensor, Iterable[Tensor]], requires_grad: bool=True
) -> Union[Tensor, Iterable[Tensor]]:
    """
    Compute the jacobian of a vector with respect to an input tensor

    :param output: a 1D vector of length L
    :param input: either a tensor (parameter) or an iterable of paramters
    :param requires_grad: whether output should be differentiable
    :return: jacobian
    """
    if isinstance(input, Tensor):
        assert len(output.shape) == 1
        grads = [grad(output[ind], input, create_graph=requires_grad)[0] for ind in range(len(output))]
        return torch.stack(grads, dim=0)
    else:
        return [compute_jacobian_v2(output, param, requires_grad) for param in input]


def update_gradient(gradients: Tensor, model: Module, fn: Callable[[Tensor], Tensor]=lambda x:x):
    for gradient, parameter in zip(gradients, model.parameters()):
        parameter.grad = fn(gradient) # Reset gradient accumulation


def accumulate_gradient(gradients, model, fn=lambda x:x):
    for gradient, parameter in zip(gradients, model.parameters()):
        parameter.grad += fn(gradient) # Reset gradient accumulation


def compute_gradient_norm(output: Tensor, model: Module):
    """
    Compute the norm of the gradient of an output (e.g a loss) with respect to a model parameters

    :param output:
    :param model:
    :return:
    """
    ret = 0
    output.backward(retain_graph=True)
    for parameter in model.parameters():
        grad = parameter.grad
        ret += grad.pow(2).sum().cpu().detach().numpy()
        parameter.grad = None # Reset gradient accumulation

    return ret


def hessian_diagonal(
        output: Tensor, input: Union[Tensor, Iterable], requires_grad: bool=True
) -> Union[Tensor, List[Tensor]]:
    """
    Compute the diagonal of the hessian

    :param output: a scalar tensor
    :param input: either a tensor (parameter), or a list/generator of parameters
    :param requires_grad: whether output should be differentiable
    :return: a tensor (parameter), or a list/generator of parameters, denoting the diagonal of hessian of output
    with respect to input
    """
    if isinstance(input, Tensor):
        original_grad = input.grad
        assert output.numel() == 1
        grads = grad(output, input, create_graph=True)[0]
        if not grads.requires_grad:
            input.grad = original_grad
            return torch.zeros(input.shape)
        grads.view(-1).backward(torch.eye(grads.numel()), create_graph=requires_grad)
        hess_diag = input.grad if input.grad is not None else torch.zeros(input.shape)
        input.grad = original_grad
        return hess_diag
    else:
        hess_diags = []
        for param in input:
            hess_diags.append(hessian_diagonal(output, param, requires_grad))
        return hess_diags


def gather_flat_grad(params: Iterable[Tensor]) -> Tensor:
    """
    Gather gradient of all the parameters and flatten into a vector. Adapted from pytorch's L-BFGS implementation.

    :param params: List of parameters
    :return: gradient vector of the parameters
    """
    views = []
    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)
