from .callbacks import Callback
from ..utils import copy_model, get_device
from typing import Dict, Any
from torch.nn import Module


__all__ = ['StochasticWeightAveraging']


class StochasticWeightAveraging(Callback):
    def __init__(
            self, learner, average_after: int,
            update_every: int=1, timescale: str="iter", device=get_device()
    ):
        """
        https://arxiv.org/pdf/1803.05407.pdf
        :param model: the model currently being trained
        :param average_after: the first epoch to start averaging
        :param update_every: how many epochs/iters between each average update
        """
        assert timescale == "epoch" or timescale == "iter"
        self.learner = learner
        self._model = learner._model
        self.model_swa = copy_model(self._model).to(device)
        self._update_every = update_every
        self._average_after = average_after
        self._timescale = timescale

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._timescale == "epoch":
            if logs["epoch"] >= self._average_after and (logs["epoch"] - self._average_after) % self._update_every == 0:
                n_model = (logs["epoch"] - self._average_after) // self._update_every
                for model_p, swa_p in zip(self._model.parameters(), self.model_swa.parameters()):
                    swa_p.data = (swa_p.data * n_model + model_p.data.to(swa_p.data.dtype)) / (n_model + 1)
                print("Update averaged model after epoch " + str(logs["epoch"]))
        return False

    def on_batch_end(self, logs: Dict[str, Any]):
        if self._timescale == "iter":
            if logs["iter_cnt"] >= self._average_after and (logs["iter_cnt"] - self._average_after) % self._update_every == 0:
                n_model = (logs["iter_cnt"] - self._average_after) // self._update_every
                for model_p, swa_p in zip(self._model.parameters(), self.model_swa.parameters()):
                    swa_p.data = (swa_p.data * n_model + model_p.data.to(swa_p.data.dtype)) / (n_model + 1)
                print("Update averaged model after iteration " + str(logs["iter_cnt"]))

    def on_train_end(self):
        self.model_swa.to(self.learner._device)
        for images, labels in self.learner._train_data:
            self.model_swa(images.to(self.learner._device))

    def get_averaged_model(self) -> Module:
        """
        Return the post-training average model
        :return: the averaged model
        """
        return self.model_swa
