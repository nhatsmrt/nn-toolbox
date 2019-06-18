from sklearn.metrics import accuracy_score
import torch

class Metric:
    def __call__(self, logs): pass

    def get_best(self) -> float: return self._best


class Accuracy(Metric):
    def __init__(self):
        self._best = 0.0

    def __call__(self, logs):
        predictions = torch.argmax(logs["outputs"], dim=1).cpu().detach().numpy()
        labels = logs["labels"].cpu().numpy()
        acc = accuracy_score(
            y_true=labels,
            y_pred=predictions
        )

        if acc >= self._best:
            self._best = acc

        return acc

class Loss(Metric):
    def __init__(self):
        self._best = float('inf')

    def __call__(self, logs):
        if logs['loss'] <= self._best:
            self._best = logs['loss']

        return logs["loss"]