from torch.utils.tensorboard import SummaryWriter
from .callbacks import Callback
from typing import Sequence


class Tensorboard(Callback):
    def __init__(self):
        self._writer = SummaryWriter()

    def on_batch_end(self, logs):
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

    def on_epoch_end(self, logs):
        if "epoch_metrics" in logs:
            for metric in logs["epoch_metrics"]:
                self._writer.add_scalar(
                    tag= "Validation " + metric,
                    scalar_value=logs["epoch_metrics"][metric],
                    global_step=logs["epoch"]
                )
        if "draw" in logs and "tag" in logs:
            print(logs["tag"])
            self._writer.add_image(
                tag=logs["tag"],
                img_tensor=logs["draw"]
            )


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

    def on_epoch_end(self, logs):
        print("Epoch " + str(logs["epoch"]) + " with:")
        for metric in self._epoch_metrics:
            assert metric in logs
            print(metric + ": " + str(logs[metric]))
