import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from math import log10
from typing import Callable
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
from typing import Tuple


__all__ = ['LRFinder']


class LRFinder:
    '''
    Leslie Smith's learning rate range finder
    Adapt from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    https://arxiv.org/pdf/1506.01186.pdf
    '''
    def __init__(
            self, model: Module, train_data: DataLoader,
            criterion: Module, optimizer: Callable[..., Optimizer], device
    ):
        self.model = model
        self.train_data = train_data
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters())
        self._device = device

    def find_lr(
            self, lr0: float=1e-7, lr_final: float=10.0, warmup: int=15,
            beta: float=0.98, verbose: bool=True, display: bool=True
    ) -> Tuple[float, float]:
        '''
        Start from a very low initial learning rate, then gradually increases it up to a big lr until loss blows up
        :param lr0: intitial learning rate
        :param lr_final: final (max) learning rate
        :param warmup: how many iterations to warmup
        :param beta: smoothing coefficient for loss
        :param verbose: whether to print out the progress
        :param display: whether to graph
        :return: a base_lr and the best lr (base_lr = best_lr / 4)
        '''
        assert warmup > 0
        model_state_dict = deepcopy(self.model.state_dict())
        lr = lr0
        num = len(self.train_data) - 1
        mult = (lr_final / lr0) ** (1 / num)
        self.optimizer.param_groups[0]['lr'] = lr

        avg_loss = 0.0
        smoothed_loss = 0.0
        iter = 0
        losses = []
        best_loss = 0.0
        log_lrs = []
        changes = []
        for inputs, labels in self.train_data:
            iter += 1
            outputs = self.model(inputs.to(self._device))
            loss = self.criterion(outputs, labels.to(self._device))
            if not torch.isnan(loss).any():
                avg_loss = beta * avg_loss + (1 - beta) * loss.cpu().item()
                changes.append(avg_loss / (1 + beta ** iter) - smoothed_loss)
                smoothed_loss = avg_loss / (1 + beta ** iter)
                losses.append(smoothed_loss)
                log_lrs.append(log10(lr))

                if verbose:
                    print("LR: " + str(lr))
                    print("Loss Change: " + str(changes[-1]))
                    print("Loss: " + str(loss.cpu().item()))
                    print("Smoothed loss: " + str(smoothed_loss))
                    print()

                if iter > warmup and smoothed_loss > best_loss * 4:
                    print("Loss blows up")
                    break
                    # return log_lrs, losses
                if smoothed_loss < best_loss or iter == warmup:
                    best_loss = smoothed_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lr *= mult
                self.optimizer.param_groups[0]['lr'] = lr
            else:
                print("Loss becomes NaN")
                break

        self.model.load_state_dict(model_state_dict)
        # logs, losses = find_lr()
        if display:
            plt.plot(log_lrs, losses)

        best_ind = np.argmin(changes)
        max_lr = 10 ** log_lrs[best_ind]
        print("Largest Loss Decrease: " + str(changes[best_ind]))
        print("Corresponding LR: " + str(max_lr))
        return max_lr / 4, max_lr
