import torch
from torch import nn
import math


class CrossLayer(nn.Module):
    """
    Implement a (residual) crossing layer for Deep and Cross Net (DCN):

    x_{l+1} = x_0 x^T_l w + b + x_l

    Based on: https://arxiv.org/pdf/1708.05123.pdf
    """
    def __init__(self, n_hidden, bias=True, return_first=False):
        super(CrossLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(1, n_hidden))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_hidden))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._return_first = return_first

    def reset_parameters(self):
        """
        Reset the parameters of the model
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, 0, 0)

    def forward(self, inputs):
        """
        :param inputs: a tuple: first element is the orinal features, second element is the output of last layer
        :return:
        """
        input, first = inputs

        interaction = torch.bmm(
            first.view(first.shape[0], first.shape[1], 1),
            input.view(input.shape[0], 1, input.shape[1])
        )
        interaction = torch.bmm(
            interaction,
            self.weight.t().view(-1, self.weight.shape[1], self.weight.shape[0]).repeat(interaction.shape[0], 1, 1)
        ).view(interaction.shape[0], interaction.shape[1])

        if self.bias is not None:
            interaction += self.bias

        if self._return_first:
            return (interaction + input, first)
        else:
            return interaction + input
