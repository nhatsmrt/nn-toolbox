from torch import Tensor
from torch.nn import Module
from .hooks import Hook


__all__ = ['InputHook', 'OutputHook']


class InputHook(Hook):
    """
    Keep this for backward compatibility
    """

    def __init__(self, module: Module, forward: bool=True):
        def store_input(hook, m, inp, op):
            hook.store = inp[0]
        super(InputHook, self).__init__(
            module=module,
            forward=forward,
            hook_func=store_input
        )


class InputHookV2(Hook):
    """
    A generic hook for storing input hook (UNTESTED)
    """
    def __init__(self, module: Module, forward: bool=True):
        super(InputHookV2, self).__init__(
            module=module,
            forward=forward,
            hook_func=self.store_input
        )

    @staticmethod
    def store_input(hook: Hook, m: Module, inp: Tensor, op: Tensor):
        hook.store = inp[0]


class OutputHook(Hook):
    """
    A generic hook for storing output hook

    A subclass would implement the store_output function (i.e decide which part of the output to store) (UNTESTED)
    """

    store: Tensor

    def __init__(self, module: Module, forward: bool=True):
        super(OutputHook, self).__init__(
            module=module,
            forward=forward,
            hook_func=self.store_output
        )

    @staticmethod
    def store_output(hook: Hook, m: Module, inp: Tensor, op: Tensor):
        hook.store = op
