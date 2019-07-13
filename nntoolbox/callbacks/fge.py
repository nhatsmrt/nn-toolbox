from .callbacks import Callback
from ..utils import copy_model
from typing import Dict, Any, List
from torch.nn import Module
from collections import deque


__all__ = ['FastGeometricEnsembling']


# UNTESTED
class FastGeometricEnsembling(Callback):
    def __init__(self, model: Module, max_n_model: int, save_after: int, save_every: int=1, timescale: str="iter"):
        """
        https://arxiv.org/pdf/1802.10026.pdf
        https://arxiv.org/pdf/1704.00109.pdf
        :param model: the model currently being trained
        :param average_after: the first epoch to start averaging
        :param update_every: how many epochs/iters between each average update
        """
        assert timescale == "epoch" or timescale == "iter"
        self._model = model
        self.models = deque()
        self._save_every = save_every
        self._save_after = save_after
        self._timescale = timescale
        self._max_n_model = max_n_model
        self.learner = None

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._timescale == "epoch":
            if logs["epoch"] >= self._save_after and (logs["epoch"] - self._save_after) % self._save_every == 0:
                self.models.append(copy_model(self._model))
                if len(self.models) > self._max_n_model:
                    self.models.pop()
                print("Save model after epoch " + str(logs["epoch"]))
        return False

    def on_batch_end(self, logs: Dict[str, Any]):
        if self._timescale == "iter":
            if logs["iter_cnt"] >= self._save_after and (logs["iter_cnt"] - self._save_after) % self._save_every == 0:
                self.models.append(copy_model(self._model))
                if len(self.models) > self._max_n_model:
                    self.models.pop()
                print("Save model after iteration " + str(logs["iter_cnt"]))

    def on_train_end(self):
        self.models = list(self.models)
        self.models = [model.to(self.learner.device) for model in self.models]

    def get_models(self) -> List[Module]:
        """
        Return the post-training average model
        :return: the averaged model
        """
        return list(self.models)
