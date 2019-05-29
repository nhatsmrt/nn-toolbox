import torch
from torch import nn

class ResidualLinearBlock(nn.Module):
    '''
    A two-layer linear block with residual connection:
    y = f(w_2f(w_1 x + b_1) + b_2) + x
    '''
    def __init__(self, in_features, activation=nn.ReLU, bias=True):
        super(ResidualLinearBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features, bias=bias),
                activation(),
                nn.Linear(in_features=in_features, out_features=in_features, bias=bias),
                activation()
            )
        )

    def forward(self, input):
        return input + self._modules["main"](input)