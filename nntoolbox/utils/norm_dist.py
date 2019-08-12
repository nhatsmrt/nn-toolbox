"""Utility functions involving computing norms and distances"""
import torch
from torch import Tensor
import torch.nn.functional as F


__all__ = ['emb_pairwise_dist', 'compute_squared_norm', 'pairwise_dist']


# Follows https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
@torch.no_grad()
def emb_pairwise_dist(embeddings: Tensor, squared: bool=True, eps: float=1e-16) -> Tensor:
    interaction = embeddings.mm(torch.t(embeddings))  # EE^T, (M, M)
    # norm = torch.norm(embeddings, dim = -1).view(embeddings.shape[0], 1)
    # sqr_norm_i = \sum_j E_{i, j}^2 = E_i E^T_i
    square_norm = torch.diag(interaction).view(embeddings.shape[0], 1)  # (M, 1)
    squared_dist = square_norm - 2 * interaction + torch.t(square_norm)
    squared_dist = F.relu(squared_dist)

    if squared:
        return squared_dist
    else:
        mask = torch.eq(squared_dist, 0).float()
        squared_dist = squared_dist + mask * eps
        dist = torch.sqrt(squared_dist)
        dist = dist * (1.0 - mask)
        return dist


@torch.no_grad()
def compute_squared_norm(A: Tensor) -> Tensor:
    """
    Compute the squared norm of each row of A

    :param A: (M, D)
    :return: squared norm (M, 1)
    """
    return torch.diag(A.mm(torch.t(A)))


@torch.no_grad()
def pairwise_dist(A: Tensor, B: Tensor) -> Tensor:
    """
    Compute pairwise distance from each row vector of A to row vector of B

    :param A: (N, D)
    :param B: (M, D)
    :return: (M, N)
    """
    sq_norm_A = compute_squared_norm(A).view(1, A.shape[0])  # (1, N)
    sq_norm_B = compute_squared_norm(B).view(B.shape[0], 1)  # (M, 1)
    interaction = B.mm(torch.t(A))  # (M, N)
    return F.relu(sq_norm_A - 2 * interaction + sq_norm_B)



