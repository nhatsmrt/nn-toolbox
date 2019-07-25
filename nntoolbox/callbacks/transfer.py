from .callbacks import Callback, GroupCallback
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, Module, Sequential
from torch.optim import Optimizer
from typing import List, Dict, Any, Optional, Union


__all__ = ['FreezeBN', 'GradualUnfreezing', 'FineTuning']
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


def unfreeze(module: Sequential, optimizer: Optimizer, unfreeze_from: int, unfreeze_to: int, **kwargs):
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
            optimizer.add_param_group({'params': submodule.parameters(), **kwargs})


class GradualUnfreezing(Callback):
    """
    Gradually unfreezing pretrained layers, with discriminative learning rates (UNTESTED)
    """
    def __init__(
            self, unfreeze_every: int, freeze_inds: Optional[List[int]]=None,
            lr: Optional[Union[List[float], float]]=None
    ):
        self._freeze_inds = freeze_inds
        self._unfreeze_every = unfreeze_every
        if lr is None:
            self.lr = None
        else:
            if isinstance(lr, list):
                assert len(lr) == len(freeze_inds)
                self.lr = lr
            else:
                self.lr = [lr for _ in range(len(freeze_inds))]

    def on_train_begin(self):
        n_layer = len(self.learner._model._modules['0'])
        if self._freeze_inds is None:
            self._freeze_inds = [n_layer - 1 - i for i in range(n_layer)]
        self._freeze_inds = [n_layer] + self._freeze_inds

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if logs['epoch'] % self._unfreeze_every == 0 \
                and logs['epoch'] > 0 \
                and logs['epoch'] // self._unfreeze_every < len(self._freeze_inds):
            unfreeze_from = self._freeze_inds[logs['epoch'] // self._unfreeze_every]
            unfreeze_to = self._freeze_inds[logs['epoch'] // self._unfreeze_every - 1]

            if self.lr is not None:
                unfreeze(
                    self.learner._model._modules['0'],
                    self.learner._optimizer,
                    unfreeze_from, unfreeze_to,
                    lr=self.lr[logs['epoch'] // self._unfreeze_every - 1]
                )
            else:
                unfreeze(
                    self.learner._model._modules['0'],
                    self.learner._optimizer,
                    unfreeze_from, unfreeze_to,
                )
            print("Unfreeze feature after " + str(unfreeze_from))
        return False


class FineTuning(GroupCallback):
    """
    Combining freezing batch norm and gradual unfreezing of layer
    """
    def __init__(self, unfreeze_every: int, freeze_inds: Optional[List[int]]=None, lr: Optional[Union[List[float], float]]=None):
        super(FineTuning, self).__init__(
            [
                GradualUnfreezing(unfreeze_every, freeze_inds, lr),
                FreezeBN()
            ]
        )
