"""Selecting pairs, triplets, npairs, etc. from a batch of data"""
# Based on https://github.com/adambielski/siamese-triplet/blob/master/utils.py
from torch import Tensor
import numpy as np
from numpy import ndarray
from itertools import combinations
from typing import Tuple
import torch
from ...utils import emb_pairwise_dist


__all__ = [
    'Selector', 'PairSelector', 'AllPairSelector',
    'TripletSelector', 'AllTripletSelector', 'BatchHardTripletSelector',
    'HardTripletSelector'
]


class Selector:
    """Abstract class for selector"""
    def select(self, embedings: Tensor, labels: Tensor) -> Tuple[Tensor, ...]: pass


class PairSelector(Selector):
    @torch.no_grad()
    def get_pairs(self, embeddings: Tensor, labels: Tensor) -> Tuple[ndarray, ndarray]:
        raise NotImplementedError

    def select(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        pos_pairs, neg_pairs = self.get_pairs(embeddings, labels)
        first = torch.cat(
            (
                torch.index_select(embeddings, dim=0, index=torch.tensor(pos_pairs[:, 0]).long().to(embeddings.device)),
                torch.index_select(embeddings, dim=0, index=torch.tensor(neg_pairs[:, 0]).long()).to(embeddings.device),
            ),
            dim=0
        )
        second = torch.cat(
            (
                torch.index_select(embeddings, dim=0, index=torch.tensor(pos_pairs[:, 1]).long().to(embeddings.device)),
                torch.index_select(embeddings, dim=0, index=torch.tensor(neg_pairs[:, 1]).long().to(embeddings.device)),
            ),
            dim=0
        )
        labels = torch.cat(
            (torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))),
            dim=0
        ).to(embeddings.device)
        return (first, second), labels


class AllPairSelector(PairSelector):
    """Select all pair from the batch"""
    @torch.no_grad()
    def get_pairs(self, embeddings: Tensor, labels: Tensor) -> Tuple[ndarray, ndarray]:
        return get_all_pairs(labels.cpu().detach().numpy())


class TripletSelector(Selector):
    @torch.no_grad()
    def get_triplets(self, embeddings: Tensor, labels: Tensor) -> ndarray:
        raise NotImplementedError

    def select(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        triplets = self.get_triplets(embeddings, labels)
        anchors = torch.index_select(embeddings, dim=0, index=torch.tensor(triplets[:, 0]).long().to(embeddings.device))
        pos = torch.index_select(embeddings, dim=0, index=torch.tensor(triplets[:, 1]).long().to(embeddings.device))
        negs = torch.index_select(embeddings, dim=0, index=torch.tensor(triplets[:, 2]).long().to(embeddings.device))
        return anchors, pos, negs


class AllTripletSelector(TripletSelector):
    @torch.no_grad()
    def get_triplets(self, embeddings: Tensor, labels: Tensor) -> ndarray:
        return get_all_triplets(labels.cpu().detach().numpy())


class BatchHardTripletSelector(TripletSelector):
    @torch.no_grad()
    def get_triplets(self, embeddings: Tensor, labels: Tensor) -> ndarray:
        return get_batch_hard_triplets(embeddings, labels.cpu().detach().numpy())


class HardTripletSelector(TripletSelector):
    def __init__(
            self, margin: float=1.0, n_neg_per_ap: int=1,
            mode: str="semi-hard"
    ):
        self._margin = margin
        self._n_neg_per_ap = n_neg_per_ap
        self._mode = mode

    @torch.no_grad()
    def get_triplets(self, embeddings: Tensor, labels: Tensor) -> ndarray:
        return get_hard_triplets(
            embeddings, labels.cpu().detach().numpy(),
            self._margin, self._n_neg_per_ap, self._mode
        )


def get_all_pairs(labels: ndarray) -> Tuple[ndarray, ndarray]:
    """Select all possible pairs from batch"""
    labels_flat = labels.ravel()
    all_pairs = np.array(list(combinations(range(len(labels_flat)), 2))).astype(np.uint8)

    pos_pairs = all_pairs[(labels_flat[all_pairs[:, 0]] == labels_flat[all_pairs[:, 1]]).astype(np.uint8).nonzero()]
    neg_pairs = all_pairs[(labels_flat[all_pairs[:, 0]] != labels_flat[all_pairs[:, 1]]).astype(np.uint8).nonzero()]

    return pos_pairs, neg_pairs


def get_all_triplets(labels: ndarray) -> ndarray:
    """Select all possible triplets of (anchor, positive, negative) from batch"""
    triplets = []
    pos_pairs, neg_pairs = get_all_pairs(labels)
    for pos in pos_pairs:
        for neg in neg_pairs:
            if pos[0] == neg[0]:
                triplets.append([pos[0], pos[1], neg[1]])
    return np.array(triplets)


def get_batch_hard_triplets(embeddings: Tensor, labels: ndarray) -> ndarray:
    """
    Implement the batch-hard strategy: For each anchor, select the corresponding hardest (furthest) positive
    and hardest (nearest) negative

    References:

    https://arxiv.org/pdf/1703.07737.pdf

    :param embeddings:
    :param labels:
    :return: array of triplet indices
    """
    triplets = []
    dist_mat = emb_pairwise_dist(embeddings, False)

    unique_class = set(labels.ravel())
    for c in unique_class:
        c_idx = np.where(labels == c)[0]
        if len(c_idx) < 2:
            continue

        other_idx = np.where(labels != c)[0]
        if len(other_idx) < 1:
            continue

        for an in c_idx:
            pos_pairs = np.array([[an, p] for p in c_idx if p != an])
            neg_pairs = np.array([[an, n] for n in other_idx])

            pos_pair_dist = dist_mat[pos_pairs[:, 0], pos_pairs[:, 1]]
            neg_pair_dist = dist_mat[neg_pairs[:, 0], neg_pairs[:, 1]]
            hardest_pos = torch.argmax(pos_pair_dist) # furthest positive
            hardest_neg = torch.argmin(neg_pair_dist) # nearest negative

            triplets.append([an, pos_pairs[hardest_pos, 1], neg_pairs[hardest_neg, 1]])

    return np.array(triplets)


def get_hard_triplets(
        embeddings: Tensor, labels: ndarray, margin: float=1.0,
        n_neg_per_ap: int=1, mode: str="semi-hard",
) -> ndarray:
    """
    Hard and semi-hard triplet selecting strategy:

    Hard: for each anchor, select negative and positive such that positive is still further to anchor than negative.

    Semi-hard: for each anchor, select negative and positive such that positive is still closer to anchor than negative,
    but the difference is less than desired margin

    :param embeddings:
    :param labels:
    :param margin:
    :param n_neg_per_ap: number of negatives to choose per anchor-positive pair
    :param mode
    :return:
    """
    triplets = []
    dist_mat = emb_pairwise_dist(embeddings, False)

    unique_class = set(labels.ravel())
    for c in unique_class:
        c_idx = np.where(labels == c)[0]
        if len(c_idx) < 2:
            continue

        other_idx = np.where(labels != c)[0]
        if len(other_idx) < 1:
            continue

        pos_pairs = np.array(list(combinations(c_idx, 2)))
        pos_pair_dist = dist_mat[pos_pairs[:, 0], pos_pairs[:, 1]]
        for pos_pair, dist in zip(pos_pairs, pos_pair_dist):
            losses = (dist - dist_mat[pos_pair[0], other_idx] + margin).cpu().detach().numpy()

            if mode == 'hard':
                hard = np.where(losses > 0)[0]
            if mode == 'semi-hard':
                hard = np.where(np.logical_and(losses > 0, losses < margin))[0]

            if len(hard) > 0:
                chosen = np.random.choice(hard, min(len(hard), n_neg_per_ap))
                neg_ind = other_idx[chosen]
                for neg in neg_ind:
                    triplets.append([pos_pair[0], pos_pair[1], neg])

    return np.array(triplets)
