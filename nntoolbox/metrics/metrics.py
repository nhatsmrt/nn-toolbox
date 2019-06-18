from sklearn.metrics import accuracy_score
import torch

class Metric:
    def __init__(self): pass

    def __call__(self, logs): pass


class Accuracy(Metric):
    def __call__(self, logs):
        predictions = torch.argmax(logs["outputs"], dim=1).cpu().detach().numpy()
        labels = logs["labels"].cpu().numpy()
        return accuracy_score(
            y_true=labels,
            y_pred=predictions
        )


class Loss(Metric):
    def __call__(self, logs):
        return logs["loss"]