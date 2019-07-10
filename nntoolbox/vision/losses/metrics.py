import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py#L20
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

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        dist = self.dist(x0, x1, squared=False)
        dist_sq = dist.pow(2)

        mdist = self.margin - dist
        cl_dist = torch.clamp(mdist, min=0.0)
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
            return torch.sqrt(dist_squared + eps)


class TripletSoftMarginLoss(nn.Module):
    def __init__(self, p = 2):
        super(TripletSoftMarginLoss, self).__init__()
        self._p = p

    def forward(self, anchor, positive, negative):
        ap = torch.norm(anchor - positive, dim = -1, p = self._p)
        an = torch.norm(anchor - negative, dim = -1,  p = self._p)
        return torch.mean(torch.log1p(torch.exp(ap - an)))


class NPairLoss(nn.Module):
    def __init__(self, reg_lambda = 0.002):
        super(NPairLoss, self).__init__()
        self._reg_lambda = reg_lambda

    # anchors, positives: (N, D)
    def forward(self, anchors, positives):
        interaction = anchors.mm(torch.t(positives)) #(N, N) (i, j) = anchor_i positive j
        labels = torch.from_numpy(np.arange(len(anchors)))

        if anchors.is_cuda:
            labels = labels.cuda()


        reg_an =  torch.mean(torch.sum(anchors * anchors, dim = -1))
        reg_pos = torch.mean(torch.sum(positives * positives, dim = -1))
        l2_reg = self._reg_lambda * 0.25 * (reg_an + reg_pos)

        return nn.CrossEntropyLoss()(interaction, labels) + l2_reg




class AngularLoss(nn.Module):
    '''
    Based on https://github.com/leeesangwon/PyTorch-Image-Retrieval/blob/public/losses.py
    '''
    def __init__(self, alpha = 45):
        super(AngularLoss, self).__init__()
        self._alpha = torch.from_numpy(np.deg2rad([alpha])).float()


    def forward(self, anchors, positives):
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
    def __init__(self, alpha = 45, reg_lambda = 0.002, angular_lambda = 2):
        super(NPairAngular, self).__init__()
        self._angular_loss = AngularLoss(alpha)
        self._npair_loss = NPairLoss(reg_lambda)
        self._angular_lambda = angular_lambda

    def forward(self, anchors, positives):
        return (self._npair_loss(anchors, positives) + self._angular_lambda * self._angular_loss(anchors, positives)) / (1 + self._angular_lambda)
