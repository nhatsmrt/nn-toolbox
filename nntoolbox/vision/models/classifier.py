from torch.nn import Module
from torch.utils.data import DataLoader

from ...utils import get_device
from ...models import Ensemble
from sklearn.metrics import accuracy_score
import torch
from torch import Tensor, nn
import numpy as np
from numpy import ndarray
from typing import List, Optional


__all__ = ['ImageClassifier', 'EnsembleImageClassifier']


class ImageClassifier:
    '''
    Abstraction for an image classifier. Support user defined test time augmentation
    '''
    def __init__(self, model: Module, tta_transform=None, tta_beta: float=0.4, device=get_device()):
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._tta_transform = tta_transform
        self._tta_beta = tta_beta
        self._softmax = nn.Softmax(dim=1)

    def predict(self, images: Tensor, return_probs: bool=False, tries: int=5) -> ndarray:
        '''
        Predict the classes or class probabilities of a batch of images
        :param images: images to be predicted
        :param return_probs: whether to return prob or classes
        :param tries: number of tries for augmentation
        :return:
        '''
        if self._tta_transform is not None:
            probs = [
                self._softmax(self._model(images.to(self._device))).cpu().detach().numpy() * self._tta_beta
            ]
            for _ in range(tries):
                transformed_images = torch.stack([self._tta_transform(image) for image in images], dim=0)
                probs.append(
                    self._softmax(
                        self._model(transformed_images.to(self._device))
                    ).cpu().detach().numpy() * (1 - self._tta_beta) / tries
                )
            probs = np.stack(probs, axis=0).sum(axis=0)
        else:
            probs = self._softmax(self._model(images.to(self._device))).cpu().detach().numpy()
        if return_probs:
            return probs
        else:
            return np.argmax(probs, axis=1)

    def evaluate(self, test_loader: DataLoader, tries: int=5) -> float:
        total = 0
        accs = 0
        for images, labels in test_loader:
            outputs = self.predict(images, tries=tries)
            labels = labels.cpu().numpy()
            acc = accuracy_score(
                y_true=labels,
                y_pred=outputs
            )
            total += len(images)
            accs += acc * len(images)
        return accs / total


class EnsembleImageClassifier(Ensemble, ImageClassifier):
    def __init__(self, models: List[ImageClassifier], model_weights: Optional[List[float]]=None):
        super(EnsembleImageClassifier, self).__init__(models, model_weights)

    def predict(self, images: Tensor, return_probs: bool=False, tries: int=5) -> ndarray:
        prediction_probs = np.stack(
            [
                self.models[i].predict(images, True, tries) * self.model_weights[i]
                for i in range(len(self.models))
            ],
            axis=0
        )
        prediction_prob = np.sum(prediction_probs, axis=0, keepdims=False)
        if return_probs:
            return prediction_prob
        else:
            return np.argmax(prediction_prob, axis=1)

