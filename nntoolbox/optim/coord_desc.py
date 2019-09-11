"""
Coordinate Descent Optimizer
"""
from torch.optim import Optimizer
from typing import List


__all__ = ['CoordDescOptimizer']


class CoordDescOptimizer(Optimizer):
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        self.ind = 0

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self, closure=None):
        self.optimizers[self.ind % len(self.optimizers)].step(closure)
        self.ind += 1
