"""Some utility functions for transfer learning"""
from torch.nn import Sequential, Module, AdaptiveAvgPool2d
from typing import Callable, Tuple


def cut_model(model: Sequential, layer_type: Callable[..., Module]=AdaptiveAvgPool2d) -> Tuple[Sequential, Sequential]:
    """
    Cut a sequential model at the first instance of layer type
    :param model:
    :param layer_type:
    :return:
    """
    cut_ind = next(i for i, o in enumerate(model.children()) if isinstance(o, layer_type))
    return model[:cut_ind], model[cut_ind:]
