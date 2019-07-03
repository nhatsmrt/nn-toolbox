"""
Implement abstraction for hooks
Adopt from FastAI:
"""
import torch
from torch.nn import Module
from typing import Callable, Any, List
from torch import Tensor
from functools import partial


__all__ = ['Hook', 'Hooks']


class Hook:
    def __init__(
            self, module: Module, hook_func: Callable[['Hook', Module, Tensor, Tensor], Any],
            forward: bool=True
    ):
        if forward:
            self.hook = module.register_forward_hook(partial(hook_func, self))
        else:
            self.hook = module.register_backward_hook(partial(hook_func, self))

    def __del__(self): self.remove()

    def remove(self): self.hook.remove()


class Hooks:
    def __init__(self, ms: List[Module], hook_fn: Callable[['Hook', Module, Tensor, Tensor], Any], forward):
        if not isinstance(forward, list):
            forward = [forward for _ in range(len(ms))]
        self.hooks = [Hook(m, hook_fn, f) for m, f in zip(ms, forward)]

    def __iter__(self): return iter(self.hooks)

    def remove(self):
        for hook in self.hooks: hook.remove()

    def __enter__(self, *args): return self

    def __exit__ (self, *args): self.remove()

    def __del__(self): self.remove()

    def __delitem__(self, i: int):
        self.hooks[i].remove()
        del(self.hooks[i])

    def __len__(self) -> int: return len(self.hooks)

    def __setitem__(self, i: int, hook: Hook): self.hooks[i] = hook
