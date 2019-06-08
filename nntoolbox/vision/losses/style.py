from torch import nn
import math
import torch

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

        return loss

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
        
        return torch.mean(torch.bmm(
            features, features.permute(0, 2, 1)
        )) / h / w / math.sqrt(n_channel)


class TotalVariationLoss(nn.Module):
    '''
    Based on https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/image_ops_impl.py
    '''
    def __init__(self, base_loss = nn.L1Loss):
        super(TotalVariationLoss, self).__init__()
        self._base_loss = base_loss()

    def forward(self, input):
        return 0.5 * (
            self._base_loss(input[:, :, 1:, :], input[:, :, :-1, :])
            + self._base_loss(input[:, :, :, 1:], input[:, :, :, :-1])
        )
