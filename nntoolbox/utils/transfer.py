"""Some utility functions for transfer learning"""
from torch.nn import Sequential, Module, AdaptiveAvgPool2d
from typing import Callable, Tuple
from torch import nn


__all__ = ['cut_sequential_model', 'cut_model']


def cut_sequential_model(
        model: Sequential, sep: Callable[..., Module]=AdaptiveAvgPool2d
) -> Tuple[Sequential, Sequential]:
    """
    Cut a sequential model at the first instance of layer type

    :param model:
    :param sep:
    :return:
    """
    cut_ind = next(i for i, o in enumerate(model.children()) if isinstance(o, sep))
    return model[:cut_ind], model[cut_ind:]


def cut_model(
        model: Sequential, sep: Callable[..., Module]=AdaptiveAvgPool2d
) -> Tuple[Sequential, Sequential]:
    """
    Cut a non-sequential model at the first instance of layer type

    :param model:
    :param sep:
    :return:
    """
    modules = [model._modules[key] for key in model._modules]
    cut_ind = [i for i in range(len(modules)) if isinstance(modules[i], sep)][0]
    return nn.Sequential(*modules[:cut_ind]), nn.Sequential(*modules[cut_ind:])
