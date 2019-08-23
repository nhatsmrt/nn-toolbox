import torch
from torch import nn, rand, Tensor
from torch.optim import Adam
from typing import Tuple


__all__ = ['all_close', 'test_component']


def all_close(t1: Tensor, t2: Tensor, eps: float=1e-6): return (t1 - t2).abs().sum() < eps


def test_component(
        component: nn.Module, inp_shape: Tuple[int, ...], op_shape: Tuple[int, ...],
        criterion: nn.Module=nn.MSELoss(), n_iter: int=1000, verbose: bool=True, eps: float=1e-6
):
    """Test if component returns expected output shape and can fit randomized input-target pair"""
    assert n_iter > 0
    inp = rand(inp_shape)
    targ = rand(op_shape)
    assert component(inp).shape == targ.shape
    optimizer = Adam(component.parameters())
    for _ in range(n_iter):
        optimizer.zero_grad()
        l = criterion(component(inp), targ)
        l.backward()
        optimizer.step()
        if verbose: print(l)
    print("Final loss: " + str(l.detach().item()))
    assert all_close(l, torch.zeros(1), eps)
