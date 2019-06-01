import torch
from torch import nn

class ResidualLinearBlock(nn.Module):
    '''
    A two-layer linear block with residual connection:
    y = f(w_2f(w_1 x + b_1) + b_2) + x
    '''
    def __init__(self, in_features, activation=nn.ReLU, bias=True, use_dropout=False, drop_rate=0.5):
        super(ResidualLinearBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features, bias=bias),
                activation(),
                nn.Dropout(drop_rate) if use_dropout else nn.Identity(),
                nn.Linear(in_features=in_features, out_features=in_features, bias=bias),
                activation()
            )
        )

    def forward(self, input):
        return input + self._modules["main"](input)


class HighwayLayer(nn.Module):
    '''
    Highway layer:
    y = T(x) * H(x) + (1 - T(x)) * x
    '''
    def __init__(self, in_features, main, gate=None):
        '''
        :param in_features: Number of features of each input
        :param main: The main network H(x). Take input of with in_features and return output with in_features
        :param gate: the gating function. Take input of with in_features and return output with in_features
        '''
        super(HighwayLayer, self).__init__()
        if gate is None:
            self._gate = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features),
                nn.Sigmoid()
            )
        else:
            self._gate = gate
        self._main = main

    def forward(self, input):
        '''
        :param input: (batch_size, in_features)
        :return: output: (batch_size, in_features)
        '''
        gate = self._gate(input)
        return gate * self._main(input) + (1 - gate) * input