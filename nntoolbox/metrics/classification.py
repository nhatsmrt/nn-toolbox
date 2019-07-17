from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from typing import Dict, Any
from torch.nn import Softmax
from torch.nn.functional import log_softmax, nll_loss
from .metrics import Metric


__all__ = ['Accuracy', 'ROCAUCScore', 'Perplexity']


class Accuracy(Metric):
    def __init__(self):
        self._best = 0.0

    def __call__(self, logs: Dict[str, Any]) -> float:
        if isinstance(logs["outputs"], torch.Tensor):
            predictions = torch.argmax(logs["outputs"], dim=1).cpu().detach().numpy()
            labels = logs["labels"].cpu().numpy()
        else:
            predictions = logs["outputs"]
            labels = logs["labels"]

        acc = accuracy_score(
            y_true=labels.ravel(),
            y_pred=predictions.ravel()
        )

        if acc >= self._best:
            self._best = acc

        return acc


class ROCAUCScore(Metric):
    def __init__(self):
        self._best = 0.0
        self.softmax = Softmax(dim=1)

    def __call__(self, logs: Dict[str, Any]) -> float:
        if isinstance(logs["outputs"], torch.Tensor):
            predictions = self.softmax(logs["outputs"]).cpu().detach().numpy()
            labels = logs["labels"].cpu().numpy()
        else:
            predictions = logs["outputs"]
            labels = logs["labels"]

        rocauc = roc_auc_score(
            y_true=labels,
            y_score=predictions[:, 1]
        )

        if rocauc >= self._best:
            self._best = rocauc

        return rocauc


class Perplexity(Metric):
    """
    Perplexity metric to evaluate a language model:

    perplexity(language_model, sentence) = exp(-log language_model(sentence))
    """
    def __init__(self):
        self._best = float('inf')

    def __call__(self, logs: Dict[str, Any]) -> float:
        labels = logs["labels"].cpu()
        predictions_prob = log_softmax(logs["outputs"], dim=1)
        entropy = nll_loss(predictions_prob, labels)
        perplexity = torch.exp(entropy).cpu().numpy().item()

        if perplexity < self._best:
            self._best = perplexity

        return perplexity
