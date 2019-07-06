from torch.optim.lr_scheduler import LambdaLR, Optimizer
from .utils import plot_schedule
from typing import Optional


class CyclicalTriangularLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, min_lr: float, max_lr: float, cycle_length: int, inc_fraction: float):
        """
        Cyclical (slanted) triangular LR, based on:
        https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules_advanced.html
        :param optimizer: pytorch optimizer
        :param min_lr: minimum learning rate
        :param max_lr: maximum learning rate
        :param cycle_length: length of each cycle (i.e from one min to another)
        :param inc_fraction: (fraction of cycle length to reach max)
        """
        assert inc_fraction > 0.0 and inc_fraction < 1.0

        def schedule_fn(iter: int) -> float:
            iter %= cycle_length
            peak_iter = int(inc_fraction * cycle_length)
            if iter <= peak_iter:
                unit_cycle = iter / cycle_length / inc_fraction
            else:
                unit_cycle = (cycle_length - iter) / cycle_length / (1 - inc_fraction)
            return unit_cycle * (max_lr - min_lr) + min_lr
        super(CyclicalTriangularLR, self).__init__(optimizer, lr_lambda=schedule_fn)
        self.iter = 0


class TriangularLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, min_lr: float, max_lr: float, cycle_length: int, inc_fraction: float):
        """
        One cycle (slanted) triangular LR, based on:
        https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules_advanced.html
        :param optimizer: pytorch optimizer
        :param min_lr: minimum learning rate
        :param max_lr: maximum learning rate
        :param cycle_length: length of each cycle (i.e from one min to another)
        :param inc_fraction: (fraction of cycle length to reach max)
        """
        assert inc_fraction > 0.0 and inc_fraction < 1.0

        def schedule_fn(iter: int) -> float:
            peak_iter = int(inc_fraction * cycle_length)
            if iter <= peak_iter:
                unit_cycle = iter / cycle_length / inc_fraction
            elif iter < cycle_length:
                unit_cycle = (cycle_length - iter) / cycle_length / (1 - inc_fraction)
            else: unit_cycle = 0.0
            return unit_cycle * (max_lr - min_lr) + min_lr

        super(TriangularLR, self).__init__(optimizer, lr_lambda=schedule_fn)
        self.iter = 0

    def step(self, iter: Optional[int] = None):
        if iter is not None:
            super().step(iter)
        else:
            self.iter += 1
            super().step(self.iter)

