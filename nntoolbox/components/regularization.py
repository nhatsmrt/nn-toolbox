from torch import nn
from typing import List, Union
import torch.nn.functional as F
import warnings


__all__ = ['DropConnect']


class DropConnect(nn.Module):
    """
    Implementation based on fastai's WeightDropout (from course 2 v3 notebook)

    Reference:

    http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf
    """

    def __init__(self, module: nn.Module, ps: Union[List[float], float]=0.0, weight_names: List[str]=['weight']):
        if isinstance(ps, list):
            assert len(ps) == len(weight_names)
        else:
            ps = [ps for _ in range(len(weight_names))]
        super(DropConnect, self).__init__()
        self.module, self.ps, self.weight_names = module, ps, weight_names
        for ind in range(len(self.weight_names)):
            weight = self.weight_names[ind]
            p = self.ps[ind]
            w = getattr(self.module, weight)

            self.register_parameter(weight + "_raw", nn.Parameter(w.data))
            self.module._parameters[weight] = F.dropout(w, p=p, training=False)

    def _setweights(self):
        for ind in range(len(self.weight_names)):
            weight = self.weight_names[ind]
            p = self.ps[ind]
            raw_w = getattr(self, weight + "_raw")

            self.module._parameters[weight] = F.dropout(raw_w, p=p, training=self.training)

    def forward(self, *inputs):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*inputs)



