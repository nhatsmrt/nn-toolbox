from torch.nn import Module
from torch.utils.data import DataLoader

from ..utils import get_device
from ..metrics import Metric, Accuracy
from .ensemble import Ensemble
from sklearn.metrics import accuracy_score
import torch
from torch import Tensor, nn
import numpy as np
from numpy import ndarray
from typing import List, Optional


__all__ = ['Classifier']


class Classifier:
    """
    Abstraction for an classifier
    """
    def __init__(self, model: Module, device=get_device(), metric: Metric=Accuracy()):
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._softmax = nn.Softmax(dim=1)
        self.metric = metric

    def predict(self, inputs: Tensor, return_probs: bool=False) -> ndarray:
        """
        Predict the classes or class probabilities of a batch of inputs
        :param inputs: inputs to be predicted
        :param return_probs: whether to return prob or classes
        :return:
        """
        probs = self._softmax(self._model(inputs.to(self._device)))
        if return_probs:
            return probs.cpu().detach().numpy()
        else:
            return torch.argmax(probs, dim=1).cpu().detach().numpy()

    def evaluate(self, test_loader: DataLoader, requires_prob: bool=False) -> float:
        total = 0
        metrics = 0
        for inputs, labels in test_loader:
            outputs = self.predict(inputs, return_probs=requires_prob)
            labels = labels.cpu().numpy()
            logs = {"outputs": outputs, "labels": labels}
            metric = self.metric(logs)
            total += len(inputs)
            metrics += metric * len(inputs)
        return metrics / total
