from .callbacks import Callback
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, Module, Sequential
from torch.optim import Optimizer
from typing import List, Dict, Any


__all__ = ['FreezeBN', 'GradualUnfreezing']
BN_TYPE = [BatchNorm1d, BatchNorm2d, BatchNorm3d]


class FreezeBN(Callback):
    """
    Freeze statistics of non trainable batch norms so that it won't accumulate statistics (UNTESTED)
    """
    def on_epoch_begin(self):
        freeze_bn(self.learner._model)


def freeze_bn(module: Module):
    for submodule in module.modules():
        for bn_type in BN_TYPE:
            if isinstance(submodule, bn_type):
                if not next(submodule.parameters()).requires_grad:
                    submodule.eval()
        # freeze_bn(submodule)


def unfreeze(module: Sequential, optimizer: Optimizer, unfreeze_from: int, unfreeze_to: int):
    """
    Unfreeze a model from ind

    :param module:
    :param optimizer
    :param unfreeze_from:
    :param unfreeze_to:
    :return:
    """
    for ind in range(len(module)):
        submodule = module._modules[str(ind)]
        if ind < unfreeze_from:
            for param in submodule.parameters():
                param.requires_grad = False
        elif ind < unfreeze_to:
            for param in submodule.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': submodule.parameters()})


class GradualUnfreezing(Callback):
    def __init__(self, freeze_inds: List[int], unfreeze_every: int):
        self._freeze_inds = freeze_inds
        self._unfreeze_every = unfreeze_every

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if logs['epoch'] % self._unfreeze_every == 0 \
                and logs['epoch'] > 0 \
                and logs['epoch'] // self._unfreeze_every < len(self._freeze_inds):
            unfreeze_from = self._freeze_inds[logs['epoch'] // self._unfreeze_every]
            unfreeze_to = self._freeze_inds[logs['epoch'] // self._unfreeze_every - 1]
            unfreeze(self.learner._model._modules['0'], self.learner._optimizer, unfreeze_from, unfreeze_to)
            print("Unfreeze feature after " + str(unfreeze_from))
        return False
