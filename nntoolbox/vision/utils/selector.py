"""Selecting pairs, triplets, npairs, etc. from a batch of data"""
# Based on https://github.com/adambielski/siamese-triplet/blob/master/utils.py
from torch import Tensor
import numpy as np
from numpy import ndarray
from itertools import combinations
from typing import Tuple, List
import torch


__all__ = ['PairSelector', 'AllPairSelector']


class PairSelector:
    def get_pairs(self, embeddings: Tensor, labels: Tensor) -> Tuple[ndarray, ndarray]:
        raise NotImplementedError

    def return_pairs(self, embeddings: Tensor, labels: Tensor):
        pos_pairs, neg_pairs = self.get_pairs(embeddings, labels)
        first = torch.cat(
            (
                torch.index_select(embeddings, dim=0, index=torch.tensor(pos_pairs[:, 0]).long()),
                torch.index_select(embeddings, dim=0, index=torch.tensor(neg_pairs[:, 0]).long()),
            ),
            dim=0
        )
        second = torch.cat(
            (
                torch.index_select(embeddings, dim=0, index=torch.tensor(pos_pairs[:, 1]).long()),
                torch.index_select(embeddings, dim=0, index=torch.tensor(neg_pairs[:, 1]).long()),
            ),
            dim=0
        )
        labels = torch.cat(
            (torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))),
            dim=0
        ).to(embeddings.device)
        return first, second, labels


class AllPairSelector(PairSelector):
    """Select all pair from the batch"""
    def get_pairs(self, embeddings: Tensor, labels: Tensor) -> Tuple[ndarray, ndarray]:
        return get_all_pairs(labels.cpu().detach().numpy())


def get_all_pairs(labels: ndarray) -> Tuple[ndarray, ndarray]:
    """Select all possible pairs from batch"""
    labels_flat = labels.ravel()
    all_pairs = np.array(list(combinations(range(len(labels_flat)), 2))).astype(np.uint8)

    pos_pairs = all_pairs[(labels_flat[all_pairs[:, 0]] == labels_flat[all_pairs[:, 1]]).astype(np.uint8).nonzero()]
    neg_pairs = all_pairs[(labels_flat[all_pairs[:, 0]] != labels_flat[all_pairs[:, 1]]).astype(np.uint8).nonzero()]

    return pos_pairs, neg_pairs
