import torch
from torch import nn
from nntoolbox.sequence.utils import get_lengths
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_dim, query_dim, max_length, return_summary):
        super(Attention, self).__init__()
        self._max_length = max_length
        self._input_dim = input_dim
        self._query_dim = query_dim
        self._return_summary = return_summary

        self._softmax = nn.Softmax(dim=0)


    def forward(self, inputs, query, mask=None):
        '''
        :param inputs: a set of vectors. Shape (seq_length, n_batch, input_dim)
        :param query: a query vector. Shape (n_batch, query_dim)
        :param mask: binary, indicating which vector is padding. Shape (seq_length, n_batch). dtype: uint8
        :return: a weighted sum of the inputs. Shape (1, n_batch, input_dim)
        '''
        attn_weights = self.compute_attn_weights(inputs, query, mask) # Shape (seq_length, n_batch)

        if self._return_summary:
            return torch.sum(attn_weights.unsqueeze(-1) * inputs, dim=0), mask
        else:
            return attn_weights.unsqueeze(-1) * inputs, mask


    def compute_attn_weights(self, inputs, query, mask=None):
        '''
        Compute the attention weights
        :param inputs: a set of vectors. Shape (seq_length, n_batch, input_dim)
        :param query: a query vector. Shape (n_batch, query_dim)
        :param mask: binary, indicating which vector is padding. Shape (seq_length, n_batch). dtype: uint8
        :return: The weights for each time step. Shape (seq_length, n_batch)
        '''
        scores = self.compute_scores(inputs, query)
        if mask is not None:
            scores[~mask] = float('-inf')

        weights = self._softmax(scores)
        return weights

    def compute_scores(self, inputs, query):
        '''
        Compute the attention scores
        :param inputs: a set of vectors. Shape (seq_length, n_batch input_dim)
        :param query: a query vector. Shape (n_batch, query_dim)
        :return: The score for each time step. Shape (seq_length, n_batch)
        '''
        raise NotImplementedError


class AdditiveAttention(Attention):
    def __init__(self, input_dim, query_dim, hidden_dim, max_length, return_summary):
        super(AdditiveAttention, self).__init__(input_dim, query_dim, max_length, return_summary)
        self._input_linear = nn.Linear(self._input_dim, hidden_dim, bias=False)
        self._query_linear = nn.Linear(self._query_dim, hidden_dim, bias=False)
        self._score_linear = nn.Linear(hidden_dim, 1)


    def compute_scores(self, inputs, query):
        '''
        Compute the additive attention scores:
        e = v^T tanh(W_x X + W_q Q)
        :param inputs: a set of vectors. Shape (seq_length, n_batch, input_dim)
        :param query: a query vector. Shape (n_batch, query_dim)
        :return: The score for each timestep. Shape (seq_length, n_batch)
        '''

        return self._score_linear(
            torch.tanh(self._input_linear(inputs) + self._query_linear(query).unsqueeze(0))
        ).squeeze(dim=-1)


class MultiplicativeAttention(Attention):
    def __init__(self, input_dim, query_dim, max_length, return_summary):
        super(MultiplicativeAttention, self).__init__(input_dim, query_dim, max_length, return_summary)
        self._bilinear = nn.Bilinear(
            in1_features=self._input_dim,
            in2_features=self._query_dim,
            out_features=1,
            bias=False
        )

    def compute_scores(self, inputs, query):
        '''
        Compute the multiplicative attention scores:
        e = xAq
        :param inputs: a set of vectors. Shape (seq_length, n_batch, input_dim)
        :param query: a query vector. Shape (n_batch, query_dim)
        :return: The score for each timestep. Shape (max_length, n_batch)
        '''
        return self._bilinear(inputs, query.unsqueeze(0).repeat(inputs.shape[0], 1, 1)).squeeze(dim=-1)


