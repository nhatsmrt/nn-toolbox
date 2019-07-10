from .callbacks import Callback
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, Module


__all__ = ['FreezeBN']
BN_TYPE = [BatchNorm1d, BatchNorm2d, BatchNorm3d]


# UNTESTED
class FreezeBN(Callback):
    """
    Freeze statistics of non trainable batch norms so that it won't accumulate statistics
    """
    def on_epoch_begin(self):
        freeze_bn(self.learner._model)


def freeze_bn(module: Module):
    for submodule in module.modules():
        for bn_type in BN_TYPE:
            if isinstance(submodule, bn_type):
                if not next(submodule.parameters()).requires_grad:
                    submodule.eval()
        freeze_bn(submodule)


# class GradualUnfreezing
