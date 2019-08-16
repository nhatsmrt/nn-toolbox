"""Implement mixture of probability distribution layers"""
import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F
from typing import List, Union, Tuple


__all__ = ['MixtureOfGaussian', 'MixtureOfExpert']


class MixtureOfGaussian(nn.Linear):
    """
    A layer that generates means, stds and mixing coefficients of a mixture of gaussian distributions.

    Used as the final layer of a mixture of (Gaussian) density network.

    Only support isotropic covariances for the components.

    References:

        Christopher Bishop. "Pattern Recognition and Machine Learning"
    """
    def __init__(self, in_features: int, out_features: int, n_dist: int, bias: bool=True):
        assert n_dist > 0 and in_features > 0 and out_features > 0
        self.n_dist = n_dist
        super(MixtureOfGaussian, self).__init__(in_features, n_dist * (2 + out_features), bias)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param input:
        :return: means, stds and mixing coefficients
        """
        features = super().forward(input)
        mixing_coeffs = F.softmax(features[:, :self.n_dist], dim=-1)
        stds = torch.exp(features[:, self.n_dist:self.n_dist * 2])
        means = features[:, self.n_dist * 2:]
        return means, stds, mixing_coeffs


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
        """
        :param input:
        :return: if return_mixture, return the mixture of expert output; else return both expert score and expert output
        (with the n_expert channel coming last)
        """
        expert_scores = self.softmax(self.gate(input))
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=-1)
        expert_scores = expert_scores.view(
            list(expert_scores.shape)[:-1] + [1 for _ in range(len(expert_outputs.shape) - len(expert_scores.shape))]
            + list(expert_scores.shape)[-1:]
        )

        if self.return_mixture:
            return torch.sum(expert_outputs * expert_scores, dim=-1)
        else:
            return expert_outputs, expert_scores

