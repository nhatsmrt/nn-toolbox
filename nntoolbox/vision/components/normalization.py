import torch
from torch import nn
from torch.nn import functional as F

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

    def forward(self, input):
        if self._style is None:
            self.set_style(input)
            return self.forward(input)
        else:
            input_reshaped = input.view(input.shape[0], input.shape[1], -1)
            input_mean = input_reshaped.mean(2).unsqueeze(-1).unsqueeze(-1)
            input_std = input_reshaped.std(2).unsqueeze(-1).unsqueeze(-1)

            style_reshaped = self._style.view(self._style.shape[0], self._style.shape[1], -1)
            style_mean = style_reshaped.mean(2).unsqueeze(-1).unsqueeze(-1)
            style_std = style_reshaped.std(2).unsqueeze(-1).unsqueeze(-1)

            return input - input_mean / (input_std + 1e-8) * style_std + style_mean

    def set_style(self, style):
        self._style = style