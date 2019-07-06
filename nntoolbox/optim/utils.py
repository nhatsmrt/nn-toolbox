import torch
from torch.optim import Optimizer
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['change_lr', 'plot_schedule']


def change_lr(optim: Optimizer, lr: float):
    for param_group in optim.param_groups:
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