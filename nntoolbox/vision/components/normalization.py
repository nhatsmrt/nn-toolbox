import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
import numpy as np


__all__ = ['L2NormalizationLayer', 'AdaIN', 'SelfStabilizer', 'BatchRenorm2D']


class L2NormalizationLayer(nn.Module):
    def __init__(self):
        super(L2NormalizationLayer, self).__init__()

    def forward(self, input):
        return F.normalize(input, dim=-1, p=2)


class AdaIN(nn.Module):
    """
    Implement adaptive instance normalization layer
    """
    def __init__(self):
        super(AdaIN, self).__init__()
        self._style = None

    def forward(self, input: Tensor) -> Tensor:
        if self._style is None:
            self.set_style(input)
            return self.forward(input)
        else:
            input_mean, input_std = AdaIN.compute_mean_std(input) # (batch_size, C, H, W)
            style_mean, style_std = AdaIN.compute_mean_std(self._style) # (batch_size, C, H, W)
            return (input - input_mean) / (input_std + 1e-8) * style_std + style_mean

    def set_style(self, style):
        self._style = style

    @staticmethod
    def compute_mean_std(images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param images: (n_img, C, H, W)
        :return: (n_img, C, H, W)
        """
        images_reshaped = images.view(images.shape[0], images.shape[1], -1)
        images_mean = images_reshaped.mean(2).unsqueeze(-1).unsqueeze(-1)
        images_std = images_reshaped.std(2).unsqueeze(-1).unsqueeze(-1)
        return images_mean, images_std


class SelfStabilizer(nn.Module):
    """
    Self stabilize layer, based on:
    https://www.cntk.ai/pythondocs/cntk.layers.blocks.html
    https://www.cntk.ai/pythondocs/_modules/cntk/layers/blocks.html#Stabilizer
    https://www.cntk.ai/pythondocs/layerref.html#batchnormalization-layernormalization-stabilizer
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/SelfLR.pdf
    """
    def __init__(self, steepness: float=4.0):
        super(SelfStabilizer, self).__init__()
        self.steepness = steepness
        self.param = nn.Parameter(torch.tensor(np.log(np.exp(steepness) - 1) / steepness))

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(self.param, beta=self.steepness) * input


class BatchRenorm2D(nn.Module):
    """
    Modified from batch norm implementation in FastAI course 2 v3's notebook. Works better for smaller batches
    (UNTESTED)

    References:

        https://github.com/fastai/course-v3/blob/master/nbs/dl2/07_batchnorm.ipynb

        Ioffe, Sergey. "Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models."
        https://arxiv.org/pdf/1702.03275.pdf

    """

    def __init__(self, num_features: int, r_max: float, d_max: float, eps: float=1e-6, momentum: float=0.1):
        assert r_max > 1.0
        assert d_max > 0.0
        assert 0.0 < momentum < 1.0

        super(BatchRenorm2D, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_var',  torch.ones(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))

        self.r_max, self.d_max = r_max, d_max
        self.eps, self.momentum = eps, momentum

    def update_stats(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_mean = input.mean((0, 2, 3), keepdim=True)
        batch_var = input.var((0, 2, 3), keepdim=True)
        batch_std = (batch_var + self.eps).sqrt()
        running_std = (self.running_var + self.eps).sqrt()

        r = torch.clamp(batch_std / running_std, min=1 / self.r_max, max=self.r_max).detach()
        d = torch.clamp((batch_mean - self.running_mean) / running_std, min=-self.d_max, max=self.d_max).detach()

        self.running_mean.lerp_(batch_mean, self.momentum)
        self.running_var.lerp_ (batch_var, self.momentum)
        return batch_mean, batch_std, r, d

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                mean, std, r, d = self.update_stats(input)
            input = (input - mean) / std * r + d
        else:
            mean, std = self.running_mean, self.running_var
            input = (input - mean) / (self.running_var + self.eps).sqrt()

        return self.weight * input + self.bias

