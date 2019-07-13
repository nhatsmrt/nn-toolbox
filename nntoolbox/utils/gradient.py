import torch
import numpy as np
from torch.nn import Module
from torch import Tensor
from typing import List, Callable


__all__ = ['compute_gradient', 'compute_jacobian', 'update_gradient', 'accumulate_gradient', 'compute_gradient_norm']


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


def compute_jacobian(input: Tensor, fn: Callable[[Tensor], Tensor], is_batch: bool=True) -> Tensor:
    """
    Compute the jacobian of function(input) with respect to input

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
        grad = [torch.autograd.grad(output[ind], [input], allow_unused=True)[0] for ind in range(len(output))]
        return torch.stack(grad, dim=0).reshape(output_shape + input_shape)


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
