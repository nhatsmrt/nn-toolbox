from torch.utils.tensorboard import SummaryWriter
from .callbacks import Callback


class Tensorboard(Callback):
    def __init__(self):
        self._writer = SummaryWriter()

    def on_batch_end(self, logs):
        self._writer.add_scalar(
            tag="Training loss",
            scalar_value=logs["loss"].item(),
            global_step=logs["iter_cnt"]
        )

    def on_epoch_end(self, logs):
        for metric in logs["epoch_metrics"]:
            self._writer.add_scalar(
                tag= "Validation " + metric,
                scalar_value=logs["epoch_metrics"][metric],
                global_step=logs["epoch"]
            )


class LossLogger(Callback):
    def __init__(self, print_every=1000):
        self._print_every = print_every

    def on_batch_end(self, logs):
        if logs["iter_cnt"] % self._print_every == 0:
            print("Iteration " + str(logs["iter_cnt"]) + ": " + str(logs["loss"]))
