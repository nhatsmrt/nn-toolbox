from torch.utils.tensorboard import SummaryWriter
from .callbacks import Callback
from typing import Sequence, Dict, Any


__all__ = ['Tensorboard', 'LossLogger', 'MultipleMetricLogger']


class Tensorboard(Callback):
    def __init__(self, every_iter: int=1, every_epoch: int=1):
        self._writer = SummaryWriter()
        self._every_iter = every_iter
        self._every_epoch = every_epoch

    def on_batch_end(self, logs):
        if logs["iter_cnt"] % self._every_iter == 0:
            if "loss" in logs:
                self._writer.add_scalar(
                    tag="Training loss",
                    scalar_value=logs["loss"].item(),
                    global_step=logs["iter_cnt"]
                )
            if "allocated_memory" in logs:
                self._writer.add_scalar(
                    tag="Allocated memory",
                    scalar_value=logs["allocated_memory"],
                    global_step=logs["iter_cnt"]
                )

    def on_epoch_end(self, logs: Dict[str, Any]):
        if logs["epoch"] % self._every_epoch == 0:
            if "epoch_metrics" in logs:
                for metric in logs["epoch_metrics"]:
                    self._writer.add_scalar(
                        tag= "Validation " + metric,
                        scalar_value=logs["epoch_metrics"][metric],
                        global_step=logs["epoch"]
                    )
            if "draw" in logs and "tag" in logs:
                for i in range(len(logs["tag"])):
                    self._writer.add_image(
                        tag=logs["tag"][i],
                        img_tensor=logs["draw"][i],
                        global_step=logs["epoch"]
                    )
        return False


class LossLogger(Callback):
    def __init__(self, print_every=1000):
        self._print_every = print_every

    def on_batch_end(self, logs):
        if logs["iter_cnt"] % self._print_every == 0:
            print("Iteration " + str(logs["iter_cnt"]) + ": " + str(logs["loss"]))


class MultipleMetricLogger(Callback):
    def __init__(self, iter_metrics: Sequence[str]=[], epoch_metrics: Sequence[str]=[], print_every=1000):
        self._print_every = print_every
        self._iter_metrics = iter_metrics
        self._epoch_metrics = epoch_metrics

    def on_batch_end(self, logs):
        if logs["iter_cnt"] % self._print_every == 0:
            print("Iteration " + str(logs["iter_cnt"]) + " with:" )
            for metric in self._iter_metrics:
                assert metric in logs
                print(metric + ": " + str(logs[metric]))

    def on_epoch_end(self, logs) -> bool:
        print("Epoch " + str(logs["epoch"]) + " with:")
        for metric in self._epoch_metrics:
            assert metric in logs
            print(metric + ": " + str(logs[metric]))
        return False
