from typing import Dict, Any, Optional
from ..utils import save_model, load_model
from ..optim.utils import save_optimizer, load_optimizer
from .callbacks import Callback


__all__ = ['ModelCheckpoint', 'OptimizerCheckPoint', 'ResumeFromCheckpoint']


class ModelCheckpoint(Callback):
    def __init__(
            self, learner, filepath: str, monitor: str='loss',
            save_best_only: bool=True, mode: str='min', period: int=1
    ):
        self._learner = learner
        self._filepath = filepath
        self._monitor = monitor
        self._period = period
        self._mode = mode
        self._save_best_only = save_best_only
        self._metrics = []

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._save_best_only:
            epoch_metrics = logs['epoch_metrics']

            assert self._monitor in epoch_metrics
            self._metrics.append(epoch_metrics[self._monitor])

            if self._mode == "min":
                if epoch_metrics[self._monitor] == min(self._metrics):
                    save_model(self._learner._model, self._filepath)
            else:
                if epoch_metrics[self._monitor] == max(self._metrics):
                    save_model(self._learner._model, self._filepath)
        else:
            save_model(self._learner._model, self._filepath)

        return False


class OptimizerCheckPoint(Callback):
    def __init__(
            self, filepath: str, monitor: str='loss',
            save_best_only: bool=True, mode: str='min', period: int=1
    ):
        self._filepath = filepath
        self._monitor = monitor
        self._period = period
        self._mode = mode
        self._save_best_only = save_best_only
        self._metrics = []

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self._save_best_only:
            epoch_metrics = logs['epoch_metrics']

            assert self._monitor in epoch_metrics
            self._metrics.append(epoch_metrics[self._monitor])

            if self._mode == "min":
                if epoch_metrics[self._monitor] == min(self._metrics):
                    save_optimizer(self.learner._optimizer, self._filepath)
            else:
                if epoch_metrics[self._monitor] == max(self._metrics):
                    save_optimizer(self.learner._optimizer, self._filepath)
        else:
            save_optimizer(self.learner._optimizer, self._filepath)

        return False


# UNTESTED
class ResumeFromCheckpoint(Callback):
    """
    Resume from previous checkpoint
    """
    def __init__(self, model_path: Optional[str]=None, optimizer_path: Optional[str]=None):
        self.model_path, self.optimizer_path = model_path, optimizer_path

    def on_train_begin(self):
        if self.model_path is not None:
            try:
                load_model(self.learner._model, self.model_path)
            except:
                print("Load model failed.")
        if self.optimizer_path is not None:
            try:
                load_optimizer(self.learner._optimizer, self.optimizer_path)
            except:
                print("Load optimizer failed.")
