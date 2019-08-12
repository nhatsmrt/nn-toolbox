from torch import nn
import torch


class Swish(nn.Module):
    """
    Swish activation function:

    f(x) = x * sigmoid(\beta x)
    """
    def __init__(self, beta_init: float=1.0, trainable: bool=True):
        super(Swish, self).__init__()
        if trainable:
            self._beta = nn.Parameter(torch.ones(1) * beta_init)
        else:
            self._beta = beta_init

    def forward(self, input):
        return input * torch.sigmoid(self._beta * input)
