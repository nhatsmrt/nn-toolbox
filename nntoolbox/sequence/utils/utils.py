from __future__ import unicode_literals, print_function, division
import unicodedata
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, nn
from typing import Union


__all__ = [
    'unicode_to_ascii', 'create_mask', 'create_mask_from_lengths',
    'get_lengths', 'extract_last', 'load_embedding'
]


def unicode_to_ascii(s: str):
    """
    Turn a Unicode string to plain ASCII

    From: https://stackoverflow.com/a/518232/2809427

    :param s: string (in unicode)
    :return: ASCII form of string
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def create_mask(inputs, pad_token):
    """
    Create a binary mask to indicate whether a token is pad or not

    :param inputs: (seq_len, batch_size)
    :param pad_token: token for padding
    :return: mask: (seq_len, batch_size)
    """
    return inputs != pad_token


def create_mask_from_lengths(inputs: Tensor, lengths: Tensor) -> Tensor:
    """
    Create a binary mask to indicate whether a token is pad or not

    :param inputs: (seq_len, batch_size)
    :param lengths: lengths of each sequence. (batch size)
    :return: mask: (seq_len, batch_size)
    """
    mask = torch.ones(size=(inputs.shape[0], inputs.shape[1])).int()
    for i in range(len(lengths)):
        mask[lengths[i]:, i] = 0
    return mask == 1


def get_lengths(mask: Union[ndarray, Tensor], return_tensor: bool=False) -> Union[ndarray, Tensor]:
    """
    Return a 1D array indicating the length of each sequence in batch

    :param mask: binary mask indicating whether an element is pad token (seq_len, batch_size)
    :param return_tensor: whether to return as a pytorch tensor
    :return: lengths (n_batch)
    """
    if return_tensor:
        return torch.sum(mask, dim=0).int()
    else:
        return np.sum(mask, axis=0).astype(np.uint8)


def extract_last(sequences: Tensor, sequence_lengths: Tensor):
    """
    Extract the last token of each padded sequence, given its length

    :param sequences: (seq_length, batch_size, n_features)
    :param sequence_lengths: (batch_size)
    :return: (batch_size, n_features)
    """
    return sequences.gather(
        dim=0,
        index=(sequence_lengths - 1).view(1, -1).unsqueeze(-1).repeat(1, 1, sequences.shape[2])
    ).squeeze(0)


def load_embedding(embedding: nn.Embedding, weight: Tensor):
    """
    Copy weight into embedding layer

    :param embedding:
    :param weight:
    """
    embedding.weight.data.copy_(weight)
