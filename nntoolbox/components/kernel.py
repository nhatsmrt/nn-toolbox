from torch import nn, Tensor
import torch


__all__ = ['DistKernel', 'GaussianDistKernel']


class DistKernel(nn.Module):
    def __call__(self, dists: Tensor) -> Tensor: pass


class GaussianDistKernel(DistKernel):
    def __init__(self, log_beta: float=0.0, trainable_beta: bool=False):
        """
        :param log_beta: log of beta (which is inverse of bandwidth)
        :param trainable_beta: whether beta should be trainable
        """
        super(GaussianDistKernel, self).__init__()
        self.log_beta = nn.Parameter(torch.tensor(log_beta), requires_grad=trainable_beta)

    def __call__(self, dists: Tensor) -> Tensor:
        return torch.exp(-torch.exp(self.log_beta) * dists.pow(2))
