from .callbacks import Callback
from ..utils import copy_model
from typing import Dict, Any, List
from torch.nn import Module


__all__ = ['FastGeometricEnsembling']


# UNTESTED
class FastGeometricEnsembling(Callback):
    def __init__(self, model: Module, average_after: int, save_every: int=1, timescale: str="iter"):
        '''
        :param model: the model currently being trained
        :param average_after: the first epoch to start averaging
        :param update_every: how many epochs/iters between each average update
        '''
        assert timescale == "epoch" or timescale == "iter"
        self._model = model
        self.models = []
        self._save_every = save_every
        self._average_after = average_after
        self._timescale = timescale

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._timescale == "epoch":
            if logs["epoch"] >= self._average_after and logs["epoch"] % self._save_every == 0:
                self.models.append(copy_model(self._model))
        return False

    def on_batch_end(self, logs: Dict[str, Any]):
        if self._timescale == "iter":
            if logs["iter_cnt"] >= self._average_after and logs["iter_cnt"] % self._save_every == 0:
                self.models.append(copy_model(self._model))

    def get_models(self) -> List[Module]:
        '''
        Return the post-training average model
        :return: the averaged model
        '''
        return self.models
