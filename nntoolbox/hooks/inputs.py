from torch.nn import Module
from .hooks import Hook


__all__ = ['InputHook']


class InputHook(Hook):
    def __init__(self, module: Module, forward: bool=True):
        def store_input(hook, m, inp, op):
            hook.store = inp[0]
        super(InputHook, self).__init__(
            module=module,
            forward=forward,
            hook_func=store_input
        )
