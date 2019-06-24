import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple

class L2NormalizationLayer(nn.Module):
    def __init__(self):
        super(L2NormalizationLayer, self).__init__()

    def forward(self, input):
        return F.normalize(input, dim=-1, p=2)


class AdaIN(nn.Module):
    '''
    Implement adaptive instance normalization layer
    '''
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
        '''
        :param images: (n_img, C, H, W)
        :return: (n_img, C, H, W)
        '''
        images_reshaped = images.view(images.shape[0], images.shape[1], -1)
        images_mean = images_reshaped.mean(2).unsqueeze(-1).unsqueeze(-1)
        images_std = images_reshaped.std(2).unsqueeze(-1).unsqueeze(-1)
        return images_mean, images_std
