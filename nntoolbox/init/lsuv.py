"""
Implement LSUV initialization from "ALL YOU NEED IS A GOOD INIT"
https://arxiv.org/pdf/1511.06422.pdf
Adopt from fastai
"""
from torch.nn import Module
from torch import Tensor,nn
from nntoolbox.hooks import Hook, OutputStatsHook
from ..utils import get_all_submodules
from torch.nn.init import orthogonal_


LINEAR_TYPE = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
__all__ = ['lsuv_init']


def lsuv_init(module: Module, input: Tensor, tol: float=1e-3, Tmax: int=100):
    """
    LSUV initialization
    :param module:
    :param input:
    :param tol: maximum tolerance
    :param Tmax: maximum iterations to attempt to demean and normalize weight
    :return: final mean and std of each layer's output
    """
    means, stds = [], []
    for layer in get_all_submodules(module):
        for type in LINEAR_TYPE:
            if isinstance(layer, type):
                orthogonal_(layer.weight) # orginal paper starts with orthogonal initialization
                hook = OutputStatsHook(layer)
                # fastai suggests demean bias as well:
                if layer.bias is not None:
                    t = 0
                    while module(input) is not None and abs(hook.stats[0][-1]) > tol and t < Tmax:
                        layer.bias.data -= hook.stats[0][-1]
                        t += 1
                if layer.weight is not None:
                    t = 0
                    while module(input) is not None and abs(hook.stats[1][-1] - 1.0) > tol and t < Tmax:
                        layer.weight.data /= hook.stats[1][-1]
                        t += 1
                hook.remove()
                means.append(hook.stats[0][-1])
                stds.append(hook.stats[1][-1])

    return means, stds
