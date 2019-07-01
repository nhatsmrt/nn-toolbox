from torch import nn


__all__ = ['ZeroCenterRelu']


class ZeroCenterRelu(nn.ReLU):
    '''
    As described by Jeremy of FastAI
    '''
    def __init__(self, inplace: bool=False):
        super(ZeroCenterRelu, self).__init__(inplace)

    def forward(self, input):
        return super().forward(input) - 0.5
