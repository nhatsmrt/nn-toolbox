"""More robust loss functions (UNTESTED)"""
from torch import nn, Tensor


__all__ = ['GeneralizedCharbonnierLoss', 'CharbonnierLoss', 'CharbonnierLossV2']


class GeneralizedCharbonnierLoss(nn.Module):
    """
    Generalized Charbonnier Loss Function:

    l(input, target) = (input - target)^2 + eps^2) ^ (alpha / 2)

    References:

        Deqing Sun et al. "Secrets of Optical Flow Estimation and Their Principles."
        http://cs.brown.edu/~dqsun/pubs/cvpr_2010_flow.pdf
    """
    def __init__(self, alpha: float=1.0, eps: float=1e-6):
        super(GeneralizedCharbonnierLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return ((input - target).pow(2) + self.eps ** 2).pow(self.alpha / 2).mean()


class CharbonnierLoss(GeneralizedCharbonnierLoss):
    """
    Charbonnier Loss Function:

    l(input, target) = sqrt((input - target)^2 + eps^2)

    References:

        Wei-Sheng Lai et al. "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks."
        https://arxiv.org/pdf/1710.01992.pdf
    """
    def __init__(self, eps: float=1e-3):
        super(CharbonnierLoss, self).__init__(1.0, eps)


class CharbonnierLossV2(nn.Module):
    """
    Charbonnier Loss Function:

    l(input, target) = sqrt((input - target)^2 + eps^2)

    References:

        Wei-Sheng Lai et al. "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks."
        https://arxiv.org/pdf/1710.01992.pdf
    """
    def __init__(self, eps: float=1e-3):
        super(CharbonnierLossV2, self).__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return ((input - target).pow(2) + self.eps ** 2).sqrt().mean()

