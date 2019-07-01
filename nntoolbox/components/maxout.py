import torch
from torch import nn, Tensor


class MaxoutLinear(nn.Module):
    '''
    A linear maxout layer
    https://arxiv.org/pdf/1302.4389.pdf
    output_i = max_{j = 1,...,k} (w_1 input + b_1, w_2 input + b_2,..., w_k input + b_k)
    '''
    def __init__(self, in_features: int, out_features: int, nb_features: int, bias: bool=True):
        super(MaxoutLinear, self).__init__()
        self._features = nn.ModuleList(
            [nn.Linear(in_features=in_features, out_features=out_features, bias=bias) for _ in range(nb_features)]
        )

    def forward(self, input: Tensor) -> Tensor:
        '''
        :param input: (batch_size, in_features)
        :return: (batch_size, out_features)
        '''
        features = [self._features[i](input) for i in range(len(self._features))]
        return torch.max(torch.stack(features, dim=-1), dim=-1)[0]
