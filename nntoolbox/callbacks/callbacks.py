from typing import Iterable, Dict, Any, List
from ..metrics import Metric
from torch import Tensor


class Callback:
    order: int=0

    def on_train_begin(self): pass

    def on_epoch_begin(self): pass

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]: return data

    def after_outputs(self, outputs: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]: return outputs

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]: return losses

    def on_backward_begin(self) -> bool: return True # if false, skip backward

    def after_backward(self) -> bool: return True # whether to continue with iteration

    def after_step(self) -> bool: return True

    # def on_phase_begin(self): pass

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool: return False # whether to stop training

    # def on_phase_end(self): pass

    def on_batch_end(self, logs: Dict[str, Any]): pass

    def on_train_end(self): pass


class GroupCallback(Callback):
    """
    Group several callbacks together (UNTESTED)
    """

    def __init__(self, callbacks: List[Callback]):
        self._callbacks = callbacks

    def on_train_begin(self):
        for cb in self._callbacks: cb.on_train_begin()

    def on_epoch_begin(self):
        for cb in self._callbacks: cb.on_epoch_begin()

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        for cb in self._callbacks:
            data = cb.on_batch_begin(data, train)
        return data

    def after_outputs(self, outputs: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        for cb in self._callbacks:
            outputs = cb.after_outputs(outputs, train)
        return outputs

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        for cb in self._callbacks:
            losses = cb.after_losses(losses, train)
        return losses

    def on_backward_begin(self) -> bool:
        ret = True
        for cb in self._callbacks: ret = ret and cb.on_backward_begin()
        return ret # if false, skip backward

    def after_backward(self) -> bool:
        ret = True
        for cb in self._callbacks: ret = ret and cb.after_backward()
        return ret # whether to continue with iteration

    def after_step(self) -> bool:
        ret = True
        for cb in self._callbacks: ret = ret and cb.after_step()
        return ret # whether to stop training

    # def on_phase_begin(self): pass

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        ret = False
        for cb in self._callbacks: ret = ret or cb.on_epoch_end(logs)
        return ret # whether to stop training

    # def on_phase_end(self): pass

    def on_batch_end(self, logs: Dict[str, Any]):
        for cb in self._callbacks: cb.on_batch_end(logs)

    def on_train_end(self):
        for cb in self._callbacks: cb.on_train_end()


class CallbackHandler:
    def __init__(
            self, learner, n_epoch: int, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, final_metric: str='accuracy'
    ):
        if metrics is not None:
            assert final_metric in metrics

        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, GroupCallback):
                    for subcb in callback._callbacks:
                        subcb.learner = learner
                        subcb.n_epoch = n_epoch
                else:
                    callback.learner = learner
                    callback.n_epoch = n_epoch

        self._callbacks = callbacks
        self._metrics = metrics
        self._final_metric = final_metric
        self._iter_cnt = 0
        self._epoch = 0
        self.learner = learner

    def on_train_begin(self):
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback.on_train_begin()

    def on_epoch_begin(self):
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback.on_epoch_begin()

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        if self._callbacks is not None:
            for callback in self._callbacks:
                data = callback.on_batch_begin(data, train)
        return data

    def after_outputs(self, outputs: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        if self._callbacks is not None:
            for callback in self._callbacks:
                outputs = callback.after_outputs(outputs, train)
        return outputs

    def after_losses(self, losses: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        if self._callbacks is not None:
            for callback in self._callbacks:
                losses = callback.after_losses(losses, train)
        return losses

    def on_backward_begin(self):
        ret = True
        if self._callbacks is not None:
            for callback in self._callbacks:
                ret = ret and callback.on_backward_begin()
        return ret

    def after_backward(self) -> bool:
        ret = True
        if self._callbacks is not None:
            for callback in self._callbacks:
                ret = ret and callback.after_backward()
        return ret

    def after_step(self) -> bool:
        ret = True
        if self._callbacks is not None:
            for callback in self._callbacks:
                ret = ret and callback.after_step()
        return ret

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
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback.on_train_end()

        if self._metrics is None:
            return 0.0
        else:
            return self._metrics[self._final_metric].get_best()


class EarlyStoppingCB(Callback):
    def __init__(self, monitor='loss', min_delta: int=0, patience: int=0, mode: str='min', baseline: float=None):
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
