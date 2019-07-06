from torch import nn, Tensor
import math
import torch
from ..components import AdaIN

__all__ = ['FeatureLoss', 'StyleLoss', 'INStatisticsMatchingStyleLoss', 'TotalVariationLoss']


class FeatureLoss(nn.Module):
    def __init__(self, model, layers, base_loss=nn.MSELoss):
        super(FeatureLoss, self).__init__()
        self._base_loss = base_loss()
        self._model = model
        self._layers = layers

    def forward(self, output, target):
        output_features, target_features = self.compute_features(output, target)

        loss = 0
        for ind in range(len(output_features)):
            loss += self._base_loss(output_features[ind], target_features[ind])

        return loss / len(self._layers)

    def compute_features(self, output, target):
        return self._model(output, self._layers),  self._model(target, self._layers)


class StyleLoss(FeatureLoss):
    def __init__(self, model, layers, base_loss=nn.MSELoss):
        super(StyleLoss, self).__init__(model, layers, base_loss)

    def compute_features(self, output, target):
        output_features = [self.gram_mat(features) for features in self._model(output, self._layers)]
        target_features = [self.gram_mat(features) for features in self._model(target, self._layers)]
        return output_features, target_features

    def gram_mat(self, features):
        batch_size = features.shape[0]
        n_channel = features.shape[1]
        h = features.shape[2]
        w = features.shape [3]
        features = features.reshape(batch_size, n_channel, -1)

        return torch.bmm(
            features, features.permute(0, 2, 1)
        ) / h / w


class INStatisticsMatchingStyleLoss(FeatureLoss):
    """
    As suggested by https://arxiv.org/pdf/1703.06868.pdf
    """
    def __init__(self, model, layers, base_loss=nn.MSELoss):
        super(INStatisticsMatchingStyleLoss, self).__init__(model, layers, base_loss)

    def compute_features(self, output, target):
        output_features = []
        target_features = []

        for feature in self._model(output, self._layers):
            mean, std = AdaIN.compute_mean_std(feature)
            output_features.append(mean)
            output_features.append(std)

        for feature in self._model(target, self._layers):
            mean, std = AdaIN.compute_mean_std(feature)
            target_features.append(mean)
            target_features.append(std)

        return output_features, target_features


class TotalVariationLoss(nn.Module):
    """
    Based on https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/image_ops_impl.py
    """
    def __init__(self, base_loss=nn.L1Loss):
        super(TotalVariationLoss, self).__init__()
        self._base_loss = base_loss()

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * (
            self._base_loss(input[:, :, 1:, :], input[:, :, :-1, :])
            + self._base_loss(input[:, :, :, 1:], input[:, :, :, :-1])
        )
