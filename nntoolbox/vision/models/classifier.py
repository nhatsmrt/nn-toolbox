from torch.nn import Module
from torch.utils.data import DataLoader

from ...utils import get_device
from ...models import Ensemble
from ...metrics import Metric

from sklearn.metrics import accuracy_score
import torch
from torch import Tensor, nn
import numpy as np
from numpy import ndarray
from typing import List, Optional, Union, Tuple, Dict, Callable

from sklearn.neighbors import KNeighborsClassifier


__all__ = ['ImageClassifier', 'KNNClassifier', 'EnsembleImageClassifier']


class ImageClassifier:
    """
    Abstraction for an image classifier. Support user defined test time augmentation
    """
    def __init__(self, model: Module, tta_transform=None, tta_beta: float=0.4, device=get_device()):
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._tta_transform = tta_transform
        self._tta_beta = tta_beta
        self._softmax = nn.Softmax(dim=1)

    def predict(self, images: Tensor, return_probs: bool=False, tries: int=5) -> ndarray:
        """
        Predict the classes or class probabilities of a batch of images

        :param images: images to be predicted
        :param return_probs: whether to return prob or classes
        :param tries: number of tries for augmentation
        :return:
        """
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


class KNNClassifier:
    def __init__(
            self, database: DataLoader, model: Module, n_neighbors: int=5,
            tta_transform=None, tta_beta: float=0.4, weights: Union[str, Callable]='distance',
            device=get_device(), threshold=0.0
    ):
        self._knn = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights
        )
        self._model = model.to(device)
        self._model.eval()

        embeddings = []
        labels = []

        for image, label in database:
            embeddings.append(self._model(image.to(device)).cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        labels = np.concatenate(labels, axis=0)

        self._knn.fit(embeddings, labels.ravel())
        self._labels_sort = np.unique(labels.ravel())
        self._n_class = np.max(labels) + 1
        self._tta_transform = tta_transform
        self._tta_beta = tta_beta
        self._device = device
        self._threshold = threshold

    def predict(
            self, images: Tensor, top: int=5, tries: int=5
    ) -> Union[ndarray, Tuple[ndarray]]:
        """
        Predict the classes or class probabilities of a batch of images
        :param images: images to be predicted
        :param tries: number of tries for augmentation
        :return:
        """
        if self._tta_transform is not None:
            embeddings = [
                self._model(images.to(self._device)).cpu().detach().numpy() * self._tta_beta
            ]
            for _ in range(tries):
                transformed_images = torch.stack([self._tta_transform(image) for image in images], dim=0)
                embeddings.append(
                    self._model(transformed_images.to(self._device)).cpu().detach().numpy() * (1 - self._tta_beta) / tries
                )
            embeddings = np.stack(embeddings, axis=0).sum(axis=0)
        else:
            embeddings = self._model(images.to(self._device)).cpu().detach().numpy()

        probs = self._knn.predict_proba(embeddings)
        probs = np.concatenate(
            (probs, np.array([[self._threshold] for _ in range(len(probs))])),
            axis=-1
        )

        best = np.argsort(-probs, axis=-1)[:, 0:top]
        best = np.array(
            [
                [
                    self._labels_sort[num] if num < len(self._labels_sort) else str(self._n_class)
                    for num in arr
                ]
                for arr in best
            ]
        )

        return self._knn.predict(embeddings), best, probs

    def evaluate(
            self, test_loader: DataLoader, metrics: Dict[str, Metric], top: int=5, tries: int=5
    ) -> Dict[str, float]:
        total = 0

        ret = {key: 0 for key in metrics}
        for images, labels in test_loader:
            outputs, best, outputs_probs = self.predict(images, top=top, tries=tries)
            labels = labels.cpu().numpy()

            batch_ret = {
                key: metrics[key]({"outputs": outputs, "best": best, "outputs_probs": outputs_probs, "labels": labels})
                for key in metrics
            }
            total += len(images)

            for key in metrics:
                ret[key] += batch_ret[key] * len(images)

        for key in metrics:
            ret[key] /= total
        return ret


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

