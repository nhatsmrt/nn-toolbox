from .callbacks import Callback
from ..utils import copy_model, get_device
from typing import Dict, Any
from torch.nn import Module


__all__ = ['LookaheadOptimizer']


class LookaheadOptimizer(Callback):
    """
    Lookahead Optimizer: Keep track of a set of "slow weights", which only update periodically. (UNTESTED)

    References:

        Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba. "Lookahead Optimizer: k steps forward, 1 step back."
        https://arxiv.org/abs/1907.08610
    """
    def __init__(
            self, step_size: float=0.5, update_every: int=1, timescale: str="iter", device=get_device()
    ):
        """
        https://arxiv.org/pdf/1803.05407.pdf
        :param model: the model currently being trained
        :param step_size: the stepsize for slow weight update
        :param average_after: the first epoch to start averaging
        :param update_every: how many epochs/iters between each average update
        """
        assert timescale == "epoch" or timescale == "iter"
        self.step_size = step_size
        self._update_every = update_every
        self._timescale = timescale
        self._device = device

    def on_train_begin(self):
        self._model = self.learner._model
        self._model_slow = copy_model(self._model).to(self._device)

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._timescale == "epoch":
            if logs["epoch"] % self._update_every == 0:
                self.update_slow_weights()
                print("Update slow weights after epoch " + str(logs["epoch"]))
        return False

    def on_batch_end(self, logs: Dict[str, Any]):
        if self._timescale == "iter":
            if logs["iter_cnt"] % self._update_every == 0:
                self.update_slow_weights()
                print("Update slow weights after iteration " + str(logs["iter_cnt"]))

    def on_train_end(self):
        self._model_slow.to(self.learner._device)
        for inputs, labels in self.learner._train_data:
            self._model_slow(inputs.to(self.learner._device))
        self.learner._model = self._model_slow

    def update_slow_weights(self):
        for model_p, slow_p in zip(self._model.parameters(), self._model_slow.parameters()):
            slow_p.data.add_(self.step_size * (model_p.data.to(slow_p.data.dtype) - slow_p.data))

    def get_final_model(self) -> Module:
        """
        Return the post-training average model
        :return: the averaged model
        """
        return self._model_slow
