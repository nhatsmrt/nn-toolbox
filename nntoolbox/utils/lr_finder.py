from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import device
from math import log10
from typing import Callable
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
from typing import Tuple, List, Optional
from .utils import is_nan


__all__ = ['LRFinder']


class LRFinder:
    """
    Leslie Smith's learning rate range finder.

    Adapt from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

    https://arxiv.org/pdf/1506.01186.pdf
    """
    def __init__(
            self, model: Module, train_data: DataLoader,
            criterion: Module, optimizer: Callable[..., Optimizer], device: device
    ):
        self.model = model
        self.train_data = train_data
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters())
        self._device = device

    def find_lr(
            self, lr0: float=1e-7, lr_final: float=10.0, warmup: int=15,
            beta: float=0.67, verbose: bool=True, display: bool=True, callbacks: Optional[List['Callback']]=None
    ) -> Tuple[float, float]:
        """
        Start from a very low initial learning rate, then gradually increases it up to a big lr until loss blows up

        :param lr0: intitial learning rate
        :param lr_final: final (max) learning rate
        :param warmup: how many iterations to warmup
        :param beta: smoothing coefficient for loss
        :param verbose: whether to print out the progress
        :param display: whether to graph
        :param callbacks: an optional list of callbacks to process input
        :return: a base_lr and the best lr (base_lr = best_lr / 4)
        """
        assert warmup > 0

        model_state_dict = deepcopy(self.model.state_dict())
        lr = lr0
        num = len(self.train_data) - 1
        mult = (lr_final / lr0) ** (1 / num)
        self.optimizer.param_groups[0]['lr'] = lr

        avg_loss = 0.0
        iter = 0
        losses = []
        best_loss = 0.0
        log_lrs = []
        changes = []
        smoothed_loss = float('-inf')

        for inputs, labels in self.train_data:
            if callbacks is None:
                outputs = self.model(inputs.to(self._device))
                loss = self.criterion(outputs, labels.to(self._device))
            else:
                data = {"inputs": inputs, "labels": labels}
                for callback in callbacks:
                    data = callback.on_batch_begin(data, True)
                inputs, labels = data["inputs"], data["labels"]
                outputs = self.model(inputs)
                for callback in callbacks:
                    outputs = callback.after_outputs({"outputs": outputs}, True)["outputs"]
                loss = self.criterion(outputs, labels.to(self._device))
                for callback in callbacks:
                    loss = callback.after_losses({"loss": loss}, True)["loss"]

            if not is_nan(loss):
                if iter == 0:
                    avg_loss = loss.cpu().item()
                else:
                    avg_loss = beta * avg_loss + (1 - beta) * loss.cpu().item()
                new_smoothed_loss = avg_loss / (1 + beta ** iter)
                changes.append(new_smoothed_loss - smoothed_loss)
                smoothed_loss = new_smoothed_loss

                losses.append(smoothed_loss)
                log_lrs.append(log10(lr))

                if verbose:
                    print("LR: " + str(lr))
                    print("Loss: " + str(loss.cpu().item()))
                    print("Smoothed loss: " + str(smoothed_loss))
                    print("Change: " + str(changes[iter]))
                    print()

                if iter > warmup and smoothed_loss > best_loss * 4:
                    print("Loss blows up")
                    break
                if smoothed_loss < best_loss or iter == warmup:
                    best_loss = smoothed_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lr *= mult
                iter += 1
                self.optimizer.param_groups[0]['lr'] = lr
            else:
                print("Loss becomes NaN")
                break

        self.model.load_state_dict(model_state_dict)
        log_lrs, losses, changes = log_lrs[5:-1], losses[5:-1], changes[5:-1]
        if display:
            plt.plot(np.power(10, log_lrs), losses)
            plt.title('LR Range Plot')
            plt.xlabel('Learning rate (log scale)')
            plt.ylabel('Losses')
            plt.show()

        # best_ind = np.argmin(losses)
        best_ind = np.argmin(changes)
        max_lr = 10 ** log_lrs[best_ind]
        # print("Minimum (smoothed) loss: " + str(losses[best_ind]))
        print("Largest change in (smoothed) loss: " + str(changes[best_ind]))
        print("Corresponding LR: " + str(max_lr))
        return max_lr / 4, max_lr


# class LRFinderV2:
#     """Adapt for any learner that has ONE optimizer and ONE loss (INCOMPLETE)"""
#     def __int__(self, learner):
#         self.learner = learner
#
#     def find_lr(
#             self, lr0: float=1e-7, lr_final: float=10.0, warmup: int=15,
#             beta: float=0.67, verbose: bool=True, display: bool=True,
#             callbacks: Optional[List['Callback']]=None
#     ):
#         callbacks += []
