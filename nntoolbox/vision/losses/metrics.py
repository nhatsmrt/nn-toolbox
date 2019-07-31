import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
from typing import Tuple
from ...components import MLP


__all__ = [
    'VerificationLoss', 'ContrastiveLoss', 'TripletSoftMarginLoss',
    'TripletMarginLossV2', 'AngularLoss', 'NPairLoss', 'NPairAngular'
]


class VerificationLoss(nn.Module):
    """
    Verify if two embeddings belong to the same class
    """

    def __init__(self, embedding_dim: int):
        super(VerificationLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.verifier = MLP(in_features=embedding_dim, out_features=1, hidden_layer_sizes=[embedding_dim // 2])
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, data: Tuple[Tensor, ...]) -> Tensor:
        (x0, x1), y = data
        assert x0.shape[-1] == x1.shape[-1] == self.embedding_dim
        assert len(x0) == len(x1) == len(y)

        y = y.float()
        if len(x0.shape) == len(y.shape) + 1:
            y = y.unsqueeze(-1)

        score = self.verifier(torch.abs(x0 - x1))
        return self.loss(score, y)

    def get_verifier(self):
        return self.verifier


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    
    Based on: 
    
    https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py#L20
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, data: Tuple[Tensor, ...]) -> Tensor:
        (x0, x1), y = data
        self.check_type_forward((x0, x1, y))
        y = y.to(x0.dtype)

        # euclidian distance
        dist = self.dist(x0, x1, squared=False)
        dist_sq = dist.pow(2)

        mdist = self.margin - dist
        cl_dist = torch.clamp(mdist, min=0.0)
        # print(dist)
        loss = y * dist_sq + (1 - y) * cl_dist.pow(2)
        loss = torch.sum(loss) / 2.0 / x0.shape[0]
        return loss

    def dist(self, x_0, x_1, eps = 1e-8, squared = False):
        interaction = x_0.mm(torch.t(x_1))
        norm_square_0 = torch.diag(x_0.mm(torch.t(x_0))).view(x_0.shape[0], 1)
        norm_square_1 = torch.diag(x_1.mm(torch.t(x_1))).view(1, x_1.shape[0])
        dist_squared = norm_square_0 - 2 * interaction + norm_square_1
        if squared:
            return dist_squared
        else:
            return torch.sqrt(torch.clamp(dist_squared, 0.0) + eps)


class TripletSoftMarginLoss(nn.Module):
    def __init__(self, p: float=2.0):
        super(TripletSoftMarginLoss, self).__init__()
        self._p = p

    def forward(self, data: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        anchor, positive, negative = data
        ap = torch.norm(anchor - positive, dim = -1, p = self._p)
        an = torch.norm(anchor - negative, dim = -1,  p = self._p)
        return torch.mean(torch.log1p(torch.exp(ap - an)))


class TripletMarginLossV2(nn.TripletMarginLoss):
    """A quick wrapper for margin loss"""
    def __init__(self, margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean'):
        super(TripletMarginLossV2, self).__init__(margin, p, eps, swap, size_average, reduce, reduction)

    def forward(self, data: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        anchor, positive, negative = data
        return super().forward(anchor, positive, negative)


class NPairLoss(nn.Module):
    def __init__(self, reg_lambda: float=0.002):
        super(NPairLoss, self).__init__()
        self._reg_lambda = reg_lambda
        self.ce_loss = nn.CrossEntropyLoss()

    # anchors, positives: (N, D)
    def forward(self, data: Tuple[Tensor, Tensor]) -> Tensor:
        anchors, positives = data
        interaction = anchors.mm(torch.t(positives)) #(N, N) (i, j) = anchor_i positive j
        labels = torch.from_numpy(np.arange(len(anchors)))

        if anchors.is_cuda:
            labels = labels.cuda()

        reg_an =  torch.mean(torch.sum(anchors * anchors, dim = -1))
        reg_pos = torch.mean(torch.sum(positives * positives, dim = -1))
        l2_reg = self._reg_lambda * 0.25 * (reg_an + reg_pos)

        return self.ce_loss(interaction, labels) + l2_reg


class AngularLoss(nn.Module):
    """
    Based on https://github.com/leeesangwon/PyTorch-Image-Retrieval/blob/public/losses.py
    """
    def __init__(self, alpha = 45):
        super(AngularLoss, self).__init__()
        self._alpha = torch.from_numpy(np.deg2rad([alpha])).float()

    def forward(self, data: Tuple[Tensor, Tensor]) -> Tensor:
        anchors, positives = data
        if anchors.is_cuda:
            self._alpha = self._alpha.cuda()

        # Normalize anchors and positives:
        anchors = F.normalize(anchors, dim = -1, p = 2)
        positives = F.normalize(positives, dim = -1, p = 2)

        n_pair = len(anchors)
        # get negative indices: (N, N - 1)
        all_pairs = np.array([[j for j in range(n_pair) if j != i] for i in range(n_pair)]).astype(np.uint8)
        stack_an = torch.stack([anchors[all_pairs[i]] for i in range(n_pair)]) # (N, N - 1, D)
        stack_pos = torch.stack([positives[all_pairs[i]] for i in range(n_pair)]) # (N, N - 1, D)
        negatives = torch.cat((stack_an, stack_pos), dim=1) # (N, 2 * (N - 1), D)

        anchors = torch.unsqueeze(anchors, dim=1)  # (N, 1, D)
        positives = torch.unsqueeze(positives, dim=1)  # (N, 1, D)
        angle_bound = torch.tan(self._alpha).pow(2)

        interaction = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (N, 1, 2 * (N - 1))

        with torch.no_grad():
            t = torch.max(interaction, dim=2)[0]

        interaction = torch.exp(interaction - t.unsqueeze(dim=1))
        interaction = torch.log(torch.exp(-t) + torch.sum(interaction, 2))
        loss = torch.mean(t + interaction)

        return loss


class NPairAngular(nn.Module):
    """
    Combining N-Pair loss and Angular loss
    """
    def __init__(self, alpha = 45, reg_lambda = 0.002, angular_lambda = 2):
        super(NPairAngular, self).__init__()
        self._angular_loss = AngularLoss(alpha)
        self._npair_loss = NPairLoss(reg_lambda)
        self._angular_lambda = angular_lambda

    def forward(self, data: Tuple[Tensor, Tensor]) -> Tensor:
        anchors, positives = data
        return (
                   self._npair_loss(anchors, positives)
                   + self._angular_lambda * self._angular_loss(anchors, positives)
               ) / (1 + self._angular_lambda)

