from typing import Callable
from sklearn.model_selection import KFold
from torch.utils.data import Subset, Dataset
from torch import nn
from ..utils import load_model
from typing import List


__all__ = ['CVEnsembler']


class CVEnsembler:
    """
    Create an ensemble of identical models, each trained on a separate (k - 1) folds of the data
    and validated on the remaining fold.

    References:

        Anders Krogh and Jesper Vedelsby. "Neural Network Ensembles, Cross Validation, and Active Learning."
        https://papers.nips.cc/paper/1001-neural-network-ensembles-cross-validation-and-active-learning.pdf
    """
    def __init__(
            self, data: Dataset, path: str, n_model: int, model_fn: Callable[..., nn.Module],
            learn_fn: Callable[[Dataset, Dataset, nn.Module, str], None]
    ):
        """
        :param data: The full dataset
        :param n_model: number of models to generated for the ensemble
        :param model_fn: a function that returns a model
        :param learn_fn: a function that takes in a train dataset, a val dataset, a model and a save path
        and save the learned model at save path
        """
        self.model_fn = model_fn
        self.n_model = n_model
        self.kf = KFold(n_splits=n_model)
        self.data = data
        self.learn_fn = learn_fn
        self.path = path

    def learn(self):
        model_ind = 0
        for train_idx, val_idx in self.kf.split(list(range(len(self.data)))):
            save_path = self.path + "model_" + str(model_ind) + ".pt"
            train_data = Subset(self.data, train_idx)
            val_data = Subset(self.data, val_idx)
            model = self.model_fn()
            self.learn_fn(train_data, val_data, model, save_path)
            model_ind += 1

    def get_models(self) -> List[nn.Module]:
        models = []
        for i in range(self.n_model):
            model = self.model_fn()
            load_path = self.path + "model_" + str(i) + ".pt"
            load_model(model, load_path)
            models.append(model)
        return models
