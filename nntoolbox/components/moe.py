"""Implement mixture-of-experts (moe) layers"""
import torch
from torch import Tensor, nn
from torch.nn import Module
from typing import List, Union, Tuple
import math


__all__ = ['MixtureOfExpert']


class MixtureOfExpert(Module):
    def __init__(self, experts: List[Module], gate: Module, return_mixture: bool=True):
        """
        :param experts: list of separate expert networks. Each must take the same input and return
         output of same dimensionality
        :param gate: take the input and output (un-normalized) score for each expert
        """
        super(MixtureOfExpert, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.softmax = nn.Softmax(dim=-1)
        self.return_mixture = return_mixture

    def forward(self, input: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        expert_score = self.softmax(self.gate(input))
        expert_output = torch.stack([expert(input) for expert in self.experts], dim=-1)
        expert_score = expert_score.view(
            list(expert_score.shape)[:-1] + [1 for _ in range(len(expert_output.shape) - len(expert_score.shape))]
            + list(expert_score.shape)[-1:]
        )

        if self.return_mixture:
            return torch.sum(expert_output * expert_score, dim=-1)
        else:
            return expert_output, expert_score


# class DiscreteGaussianMOE(MixtureOfExpert):
#     def __init__(self, experts: List[Module], gate: Module):
#         super(DiscreteGaussianMOE, self).__init__(experts, gate, False)
#
#     def forward(self, input: Tensor) -> Tensor:
#         # (batch_size, n_expert, output_dim), (batch_size, n_expert)
#         expert_output, expert_score = super().forward(input)
#         targets = torch.eye(expert_output.shape[2])[None, None, :] # (1, 1, output_dim, output_dim)
#         expert_output = expert_output.unsqueeze(2) # (batch_size, n_expert, 1, output_dim)
#         # (batch_size, n_expert, output_dim):
#         output = torch.exp(-(expert_output - targets).pow(2).sum(-1)) / math.sqrt(2 * math.pi)
#         output = (output * expert_score.unsqueeze(-1)).sum(1) # (batch_size, output_dim):
#         return output

