import torch
from torch import nn
from nntoolbox.sequence.utils import create_mask_from_lengths
import numpy as np
from torch import Tensor
from typing import Tuple, Any


class Attention(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, value_dim: int, return_summary: bool):
        super(Attention, self).__init__()
        self._value_dim = value_dim
        self._query_dim = query_dim
        self._key_dim = key_dim
        self._return_summary = return_summary

        self._softmax = nn.Softmax(dim=1)

    def forward(self, keys: Tensor, queries: Tensor, values: Tensor, mask=None) -> Tuple[Tensor, Any]:
        """
        :param keys: a set of vectors with values' info, to compute attention weights. (seq_length, n_batch, keys_dim)
        :param queries: queries vector. Shape (n_query, n_batch, query_dim)
        :param values: a set of vectors to be attended to. Shape (seq_length, n_batch, input_dim)
        :param mask: binary, indicating which vector is padding. Shape (seq_length, n_batch). dtype: uint8
        :return: a weighted sum of the inputs. (n_query, n_batch, input_dim) if return_summary, else (n_query, seq_length, n_batch)
        """
        attn_weights = self.compute_attn_weights(keys, queries, mask) # Shape (n_query, seq_length, n_batch, 1)

        if self._return_summary:
            return torch.sum(attn_weights * values.unsqueeze(0), dim=1), mask
        else:
            return attn_weights * values.unsqueeze(0), mask

    def compute_attn_weights(self, keys: Tensor, queries: Tensor, mask=None) -> Tensor:
        """
        Compute the attention weights

        :param keys: a set of vectors with values' info, to compute attention weights. (seq_length, n_batch, keys_dim)
        :param queries: query vectors. Shape (n_query, n_batch, query_dim)
        :param mask: binary, indicating which vector is padding. Shape (seq_length, n_batch). dtype: uint8
        :return: The weights for each time step. Shape (n_query, seq_length, n_batch, 1)
        """
        scores = self.compute_scores(keys, queries) # (n_query, seq_length, n_batch, 1)
        if mask is not None:
            scores[np.logical_not(mask.unsqueeze(0).unsqueeze(-1).repeat(scores.shape[0], 1, 1, 1))] = float('-inf')

        weights = self._softmax(scores)
        return weights

    def compute_scores(self, keys: Tensor, queries: Tensor) -> Tensor:
        """
        Compute the attention scores

        :param keys: a set of vectors with values' info, to compute attention weights. (seq_length, n_batch, keys_dim)
        :param queries: query vectors. Shape (n_query, n_batch, query_dim)
        :return: The score for each time step. Shape (n_query, seq_length, n_batch, 1)
        """
        raise NotImplementedError


class AdditiveAttention(Attention):
    def __init__(self, key_dim: int, query_dim: int, value_dim: int, hidden_dim: int, return_summary: bool):
        super(AdditiveAttention, self).__init__(key_dim, query_dim, value_dim, return_summary)
        self._key_linear = nn.Linear(self._key_dim, hidden_dim, bias=False)
        self._query_linear = nn.Linear(self._query_dim, hidden_dim, bias=False)
        self._score_linear = nn.Linear(hidden_dim, 1)

    def compute_scores(self, keys, queries):
        """
        Compute the attention scores

        :param keys: a set of vectors with values' info, to compute attention weights. (seq_length, n_batch, keys_dim)
        :param queries: query vectors. Shape (n_query, n_batch, query_dim)
        :return: The score for each time step. Shape (n_query, seq_length, n_batch, 1)
        """
        return self._score_linear(
            torch.tanh(self._key_linear(keys).unsqueeze(0) + self._query_linear(queries).unsqueeze(1))
        )


class MultiplicativeAttention(Attention):
    def __init__(self, key_dim, query_dim, value_dim, return_summary):
        super(MultiplicativeAttention, self).__init__(key_dim, query_dim, value_dim, return_summary)
        self._bilinear = nn.Bilinear(
            in1_features=self._key_dim,
            in2_features=self._query_dim,
            out_features=1,
            bias=False
        )

    def compute_scores(self, keys, queries):
        """
        Compute the attention scores
        :param keys: a set of vectors with values' info, to compute attention weights. (seq_length, n_batch, keys_dim)
        :param queries: query vectors. Shape (n_query, n_batch, query_dim)
        :return: The score for each time step. Shape (n_query, seq_length, n_batch, 1)
        """
        return self._bilinear(
            keys.unsqueeze(0).repeat(queries.shape[0], 1, 1, 1),
            queries.unsqueeze(1).repeat(1, keys.shape[0], 1, 1)
        )


class ScaledDotProductAttention(Attention):
    def __init__(self, key_dim, query_dim, value_dim, return_summary):
        assert key_dim == query_dim
        super(ScaledDotProductAttention, self).__init__(key_dim, key_dim, value_dim, return_summary)

    def compute_scores(self, keys, queries):
        """
        Compute the attention scores:
        score(K, Q) = KQ^T / sqrt(d_k)
        :param keys: a set of vectors with values' info, to compute attention weights. (seq_length, batch_size, key_dim)
        :param queries: query vectors. Shape (n_query, batch_size, query_dim = key_dim)
        :return: The score for each time step. Shape (n_query, seq_length, n_batch, 1)
        """
        return queries.permute(1, 0, 2).bmm(keys.permute(1, 2, 0)).permute(1, 2, 0).unsqueeze(-1) / (self._key_dim ** 0.5)


class SelfAttention(nn.Module):
    def __init__(
            self, base_attention, in_features: int, key_dim: int, query_dim: int, value_dim: int,
            value_as_key: bool=False, transform: bool=True, **kwargs
    ):
        super(SelfAttention, self).__init__()
        self._base_attention = base_attention(key_dim=key_dim, query_dim=query_dim, value_dim=value_dim, **kwargs)
        self._transform = transform

        if transform:
            if not value_as_key:
                self._key_linear = nn.Linear(in_features=in_features, out_features=key_dim)
            else:
                assert key_dim == value_dim
            self._value_as_key = value_as_key
            self._query_linear = nn.Linear(in_features=in_features, out_features=query_dim)
            self._value_linear = nn.Linear(in_features=in_features, out_features=value_dim)
        else:
            assert in_features == key_dim
            assert in_features == query_dim
            assert in_features == value_dim

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tuple[Tensor, Any]:
        """
        :param inputs: (seq_length, batch_size, input_dim)
        :param lengths: (batch_size)
        :return: (seq_length, batch_size, input_dim) and mask (seq_len, batch_size)
        """
        mask = create_mask_from_lengths(inputs, lengths)

        if self._transform:
            values = self._value_linear(inputs)
            keys = values if self._value_as_key else self._key_linear(inputs)
            queries = self._query_linear(inputs)
        else:
            values = keys = queries = inputs

        return self._base_attention(
            keys=keys,
            queries=queries,
            values=values,
            mask=mask
        )
