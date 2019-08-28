import math
from torch.nn import init, Module


__all__ = ['sqrt_uniform_init']


def sqrt_uniform_init(component: Module):
    for weight in component.parameters():
        stdv = 1.0 / math.sqrt(weight.shape[-1])
        init.uniform_(weight, -stdv, stdv)
