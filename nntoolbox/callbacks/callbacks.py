from typing import Iterable, Dict, Any, Tuple
from ..utils import save_model
from ..metrics import Metric
from torch import Tensor


class Callback:
    def on_train_begin(self): pass

    def on_batch_begin(self, images, labels, train) -> Tuple[Tensor, Tensor]: pass

    def on_phase_begin(self): pass

    def on_epoch_end(self, logs) -> bool: return False

    def on_phase_end(self): pass

    def on_batch_end(self, logs: Dict[str, Any]): pass

    # def on_train_end(self): pass


class CallbackHandler:
    def __init__(
            self, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, final_metric: str='accuracy'
    ):
        if metrics is not None:
            assert final_metric in metrics

        self._callbacks = callbacks
        self._metrics = metrics
        self._final_metric = final_metric
        self._iter_cnt = 0
        self._epoch = 0

    def on_batch_end(self, logs: Dict[str, Any]):
        logs["iter_cnt"] = self._iter_cnt
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback.on_batch_end(logs)

        self._iter_cnt += 1

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        print("Evaluate for epoch " + str(self._epoch) + ": ")
        logs["epoch"] = self._epoch
        stop_training = False
        if self._metrics is not None:
            epoch_metrics = dict()
            for metric in self._metrics:
                epoch_metrics[metric] = self._metrics[metric](logs)
                print(metric + ": " + str(epoch_metrics[metric]))
            logs["epoch_metrics"] = epoch_metrics

        if self._callbacks is not None:
            for callback in self._callbacks:
                stop_training = stop_training or callback.on_epoch_end(logs)

        self._epoch += 1
        return stop_training

    def on_train_end(self) -> float:
        if self._metrics is None:
            return 0.0
        else:
            return self._metrics[self._final_metric].get_best()


class ModelCheckpoint(Callback):
    def __init__(
            self, learner, filepath: str, monitor: str='loss', save_best_only=True, mode: str='min', period: int=1
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


class EarlyStoppingCB(Callback):
    def __init__(self, monitor='loss', min_delta: int=0, patience: int=0, mode: str='min', baseline=None):
        self._monitor = monitor
        self._min_delta = min_delta
        self._patience = patience
        self._cur_p = 0
        self._mode = mode
        self._baseline = baseline
        self._metrics = []

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        epoch_metrics = logs['epoch_metrics']

        assert self._monitor in epoch_metrics
        self._metrics.append(epoch_metrics[self._monitor])

        if self._mode == "min":
            if epoch_metrics[self._monitor] == min(self._metrics) and \
                    (self._baseline is None or epoch_metrics[self._monitor] <= self._baseline):
                self._cur_p = 0
            else:
                self._cur_p += 1
        else:
            if epoch_metrics[self._monitor] == max(self._metrics) and \
                    (self._baseline is None or epoch_metrics[self._monitor] >= self._baseline):
                self._cur_p = 0
            else:
                self._cur_p += 1

        return self._cur_p > self._patience
