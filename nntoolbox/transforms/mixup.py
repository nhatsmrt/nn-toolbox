import numpy as np
from torch import Tensor
from typing import Tuple
import torch


class MixupTransformer:
    def __init__(self, alpha: float=0.4):
        self._alpha = alpha

    def transform_data(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        lambd = np.random.beta(self._alpha, self._alpha, labels.size(0))
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lambd = torch.from_numpy(lambd).float().to(images.device)
        shuffle = torch.randperm(labels.size(0)).to(images.device)
        images_shuffled,labels_shuffled = images[shuffle], labels[shuffle]

        new_images = (
            images * lambd.view((lambd.size(0), 1, 1, 1))
            + images_shuffled * (1 - lambd).view((lambd.size(0), 1, 1, 1))
        )
        new_labels = torch.cat([labels[:, None].float(), labels_shuffled[:, None].float(), lambd[:, None].float()], 1)

        return new_images, new_labels

    def transform_loss(self, criterion):
        def transformed_loss(outputs, labels):
            loss1, loss2 = criterion(outputs, labels[:, 0].long()), criterion(outputs, labels[:, 1].long())
            return (loss1 * labels[:, 2] + loss2 * (1 - labels[:, 2])).mean()
        return transformed_loss
