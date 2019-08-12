import torch
from torch import nn, Tensor
import torch.nn.functional as F


__all__ = ['O2PLayer', 'O2PLayerV2', 'BilinearPooling']


def mat_log_sym(matrix: Tensor, epsilon: float=1e-10):
    """
    Compute matrix logarithm of a symmetric matrix

    :param matrix: square, symmetric matrix
    :param epsilon: 
    :return: 
    """
    U, S, V = torch.svd(
        matrix + epsilon * torch.eye(n = matrix.shape[0], m = matrix.shape[1]),
        some = False
    )

    S = torch.log(S)
    Sigma = torch.zeros(size = (U.shape[0], V.shape[0]))
    Sigma[torch.arange(S.shape[0]), torch.arange(S.shape[0])] = S

    return U.mm(Sigma.mm(torch.t(V)))


def mat_dot_log_sym(matrix: Tensor, epsilon: float=1e-10):
    """
    Matrix logarithm

    :param matrix: C X D matrix
    :param epsilon:
    :return:
    """
    U, S, V = torch.svd(matrix.t(), some = False) # U: D X D, V: C X C, S: min(D, C)

    Sigma = torch.zeros(size = (U.shape[0], V.shape[0])) # D X C
    Sigma[torch.arange(S.shape[0]), torch.arange(S.shape[0])] = S

    Sigma = mat_log_sym(Sigma.t().mm(Sigma), epsilon) # C X C
    # print(S)
    return V.mm(Sigma.mm(torch.t(V)))


class O2PLayer(nn.Module):
    """"""
    def __init__(self):
        super(O2PLayer, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        input = input.view(input.shape[0], input.shape[1], -1)
        interaction = input.bmm(input.permute(0, 2, 1))
        log_interactions = [mat_log_sym(matrix) for matrix in torch.unbind(interaction, dim=0)]

        return torch.stack(log_interactions, dim = 0)


class O2PLayerV2(nn.Module):
    """
    Slightly more stable
    Recommended Use.
    """
    def __init__(self):
        super(O2PLayerV2, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        input = input.view(input.shape[0], input.shape[1], -1)
        log_interactions = [mat_dot_log_sym(matrix) for matrix in torch.unbind(input, dim=0)]

        return torch.stack(log_interactions, dim=0)


class BilinearPooling(nn.Module):
    """
    Bilinear pooling layer

    References:

        Lin et al. "Bilinear CNN Models for Fine-grained Visual Recognition".
        http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
    """
    def __init__(self):
        super(BilinearPooling, self).__init__()

    def forward(self, inputA: Tensor, inputB: Tensor) -> Tensor:
        inputA = inputA.view(inputA.shape[0], inputA.shape[1], -1)
        inputB = inputB.view(inputB.shape[0], inputB.shape[1], -1)

        bi_vec = inputA.bmm(inputB.permute(0, 2, 1))
        bi_vec = bi_vec.view(bi_vec.shape[0], -1)
        bi_vec = torch.sign(bi_vec) * torch.sqrt(torch.abs(bi_vec))

        return F.normalize(bi_vec, dim=-1, p=2)
