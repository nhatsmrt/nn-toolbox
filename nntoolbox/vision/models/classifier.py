from torch.nn import Module
from torch.utils.data import DataLoader

from ...utils import get_device
from sklearn.metrics import accuracy_score
import torch
from torch import Tensor
from numpy import ndarray


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

    def predict(self, images: Tensor, return_probs: bool=False, tries: int=5) -> ndarray:
        '''
        Predict the classes or class probabilities of a batch of images
        :param images: images to be predicted
        :param return_probs: whether to return prob or classes
        :param tries: number of tries for augmentation
        :return:
        '''
        if self._tta_transform is not None:
            probs = [self._model(images.to(self._device)) * self._tta_beta]
            for _ in range(tries):
                transformed_images = torch.stack([self._tta_transform(image) for image in images], dim=0)
                probs.append(
                    self._model(transformed_images.to(self._device)) * (1 - self._tta_beta) / tries
                )
            probs = torch.stack(probs, dim=0).sum(dim=0)
        else:
            probs = self._model(images.to(self._device))
        if return_probs:
            return probs.cpu().detach().numpy()
        else:
            return torch.argmax(probs, dim=1).cpu().detach().numpy()

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
