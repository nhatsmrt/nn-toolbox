from typing import Optional
from torch import nn, Tensor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from .kernel import DistKernel
import torch


__all__ = ['RBFLayer']


class RBFLayer(nn.Linear):
    """
    RBF Layer (used for output or as the hidden layer for RBF network)

    References:

    Lecun et al. "Gradient-Based Learning Applied to Document Recognition."
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """
    def __init__(
            self, in_features: int, out_features: int, trainable_centers: bool=True,
            normalized: bool= False, kernel: Optional[DistKernel]=None, initial_centers: Optional[Tensor]=None
    ):
        """
        :param in_features: dimension of input
        :param out_features: number of centers
        :param trainable_centers: whether the center can be moved
        :param normalized: whether the output should be normalized (i.e sum to 1)
        :param kernel: (optional) a distance-based kernel function
        :param initial_centers: (optional) initial centers placement
        """
        super(RBFLayer, self).__init__(in_features, out_features, False)
        self.normalized = normalized
        self.kernel = kernel

        if initial_centers is not None:
            assert initial_centers.shape[0] == out_features and initial_centers.shape[1] == in_features
            self.weight = self.centers = nn.Parameter(initial_centers, requires_grad=trainable_centers)
        else:
            self.centers = self.weight
            self.centers.requires_grad = trainable_centers

    def cluster_initialize(self, input: Tensor):
        """
        (Re-)initialize the centers based on k-mean clustering on the input

        :param input:
        """
        model = KMeans(self.out_features)
        model.fit(input.cpu().detach().numpy())
        self.weight.data.copy_(torch.Tensor(model.cluster_centers_).to(self.weight.data.device))

    def centroids_initialize(self, input: Tensor, labels: Tensor):
        """
        (Re-)initialize the centers based on nearest centroids algorithm

        :param input:
        :param labels:
        """
        model = NearestCentroid()
        model.fit(input.cpu().detach().numpy(), labels.cpu().detach().numpy().ravel())
        self.weight.data.copy_(torch.Tensor(model.centroids_).to(self.weight.data.device))

    def forward(self, input: Tensor) -> Tensor:
        dists = pairwise_dist(input, self.centers, squared=True)
        if self.kernel is not None:
            dists = self.kernel(torch.sqrt(dists))
        return dists / dists.sum(dim=-1, keepdim=True) if self.normalized else dists


def pairwise_dist(A: Tensor, B: Tensor, squared: bool=True) -> Tensor:
    """
    :param A: (M, D)
    :param B: (N, D)
    :param squared: whether to return squared distance or just distance
    :return: (M, N)
    """
    interaction = A.matmul(B.t())
    A_sq = A.pow(2).sum(-1).unsqueeze(-1)
    B_sq = B.pow(2).sum(-1).unsqueeze(0)
    dist_sq = torch.clamp(A_sq + B_sq - 2 * interaction, min=0)
    return dist_sq if squared else dist_sq.sqrt()

