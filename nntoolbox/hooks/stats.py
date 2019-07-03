from .hooks import Hooks, Hook
from torch.nn import Module
from torch import Tensor
from typing import List


__all__ = ['OutputStatsHooks', 'OutputStatsHook']


class OutputStatsHook(Hook):
    def __init__(self, module: Module):
        super(OutputStatsHook, self).__init__(module, get_output_stats, True)


class OutputStatsHooks(Hooks):
    def __init__(self, ms: List[Module]):
        super(OutputStatsHooks, self).__init__(ms, get_output_stats, True)


def get_output_stats(hook: Hook, module: Module, input: Tensor, output: Tensor):
    if not hasattr(hook, 'stats'): hook.stats = ([], [])
    means, stds = hook.stats
    if module.training:
        means.append(output.data.mean().cpu().detach().item())
        stds.append(output.data.std().cpu().detach().item())
