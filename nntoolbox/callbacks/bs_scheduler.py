from .callbacks import Callback
from typing import Dict, Any, Callable
from torch.utils.data import DataLoader


__all__ = ['BatchSizeScheduler']


class BatchSizeScheduler(Callback):
    def __init__(self, train_data: DataLoader, bs_schedule_fn: Callable[[int], int], timescale: str="iter"):
        assert timescale == "iter" or timescale == "epoch"
        self.timescale = timescale
        self._train_data = train_data
        self._bs_schedule_fn = bs_schedule_fn

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self.timescale == "epoch":
            new_bs = self._bs_schedule_fn(logs["epoch"])
            self._train_data.batch_size = new_bs
            self._train_data.batch_sampler.batch_size = new_bs
        return False

    def on_batch_end(self, logs: Dict[str, Any]):
        if self.timescale == "iter":
            new_bs = self._bs_schedule_fn(logs["iter_cnt"])
            self._train_data.batch_size = new_bs
            self._train_data.batch_sampler.batch_size = new_bs


# UNTESTED
class BatchSizeIncreaser(Callback):
    """
    Implement a callback to increase batch size during training
    
    https://arxiv.org/pdf/1711.00489.pdf
    """
    def __init__(
            self, train_data: DataLoader, update_after: int, update_every: int,
            bs_init: int, lr_scheduler, bs_max: int, bs_inc_rate: float=5.0
    ):
        self._update_after = update_after
        self._update_every = update_every
        self._train_data = train_data
        self.bs_inc_rate = bs_inc_rate
        self.bs_max = bs_max
        self.cur_bs = bs_init
        self._lr_scheduler = lr_scheduler

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if logs['epoch'] >= self._update_after and (logs['epoch'] - self._update_after) % self._update_every == 0:
            new_bz = int(self.cur_bs * self.bs_inc_rate)
            if new_bz < self.bs_max:
                self.cur_bs = new_bz
                self._train_data.batch_size = new_bz
                self._train_data.batch_sampler.batch_size = new_bz
            else:
                self._lr_scheduler.step()

            print("Increase batch size to " + str(new_bz))
        return False
