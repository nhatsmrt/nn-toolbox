from torch.optim import Optimizer
from typing import Callable, Union, List
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['change_lr', 'plot_schedule']


# UNTESTED
def change_lr(optim: Optimizer, lrs: Union[float, List[float]]):
    """
    Change the learning rate of an optimizer

    :param optim: optimizer
    :param lrs: target learning rate
    """
    if isinstance(lrs, list):
        assert len(lrs) == len(optim.param_groups)
    else:
        lrs = [lrs for _ in range(len(optim.param_groups))]

    for param_group, lr in zip(optim.param_groups, lrs):
        param_group['lr'] = lr


def plot_schedule(schedule_fn: Callable[[int], float], iterations: int=30):
    """
    Plot the learning rate schedule function

    :param schedule_fn: a function that returns a learning rate given an iteration
    :param iterations: maximum number of iterations (or epochs)
    :return:
    """
    iterations = np.arange(iterations)
    lrs = np.array(list(map(schedule_fn, iterations)))
    plt.plot(iterations, lrs)
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.show()
