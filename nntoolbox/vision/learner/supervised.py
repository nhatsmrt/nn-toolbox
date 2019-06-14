from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ...utils import save_model, load_model, get_device
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class SupervisedImageLearner:

    def __init__(
            self, train_data:DataLoader, val_data:DataLoader, model:Module, criterion:Module,
            optimizer:Optimizer, val_metric=None, use_scheduler=False, use_tb=False, device=get_device()
    ):
        self._train_data = train_data
        self._val_data = val_data
        self._model = model.to(device)
        self._criterion = criterion.to(device)
        self._optimizer = optimizer
        self._val_metric = val_metric
        self._device = device

        if use_tb:
            self._writer = SummaryWriter()
        else:
            self._writer = None

        if use_scheduler:
            self._lr_scheduler = ReduceLROnPlateau(self._optimizer, mode='max')
        else:
            self._lr_scheduler = None

    def learn(
            self, n_epoch:int, print_every:int, eval_every:int=1,
            load_path=None, save_path=None, patience=None, verbose=True
    ) -> float:
        if load_path is not None:
            load_model(self._model, load_path)

        iter_cnt = 0
        p = 0
        val_metrics = []
        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._model.train()

            for images, labels in self._train_data:
                loss = self.learn_one_iter(images, labels)
                if iter_cnt % print_every == 0 and verbose:
                    print(loss)

                if self._writer is not None:
                    self._writer.add_scalar(
                        tag="Training loss",
                        scalar_value=loss.item(),
                        global_step=iter_cnt
                    )

                iter_cnt += 1

            if e % eval_every == 0:
                print("Evaluate: ")
                val_metric = self.evaluate()
                print(self._val_metric + ": " + str(val_metric))
                val_metrics.append(val_metric)
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step(val_metric)

                if self.is_best(val_metric, val_metrics):
                    p = 0
                    if save_path is not None:
                        save_model(self._model, save_path)
                else:
                    p += 1

                if self._writer is not None:
                    self._writer.add_scalar(
                        tag="Val " + self._val_metric,
                        scalar_value=val_metric,
                        global_step=e
                    )

                if patience is not None and p > patience:
                    print("Patience exceeded. Finish training.")
                    break

        self._writer.close()
        return max(val_metrics) if self._val_metric == 'accuracy' else min(val_metrics)

    def learn_one_iter(self, images, labels):
        self._optimizer.zero_grad()
        loss = self.compute_loss(images.to(self._device), labels.to(self._device))
        loss.backward()
        self._optimizer.step()
        return loss

    @torch.no_grad()
    def evaluate(self) -> float:
        self._model.eval()
        vals = []
        total_data = 0

        for images, labels in self._val_data:
            if self._val_metric is None or self._val_metric == 'loss':
                val_metric = self.compute_loss(images, labels).item()
            elif self._val_metric == 'accuracy':
                val_metric = self.compute_accuracy(images, labels).item()
            else:
                raise not NotImplementedError

            total_data += len(images)
            vals.append(val_metric * len(images))

        return np.sum(vals) / total_data

    def compute_loss(self, images, labels):
        return self._criterion(self._model(images), labels)

    @torch.no_grad()
    def compute_accuracy(self, images, labels) -> float:
        outputs = torch.argmax(self._model(images.to(self._device)), dim=1).cpu().detach().numpy()
        labels = labels.cpu().numpy()
        return accuracy_score(
            y_true=labels,
            y_pred=outputs
        )

    def is_best(self, val, vals) -> bool:
        if self._val_metric == 'loss':
            return val == np.min(vals)
        elif self._val_metric == 'accuracy':
            return val == np.max(vals)
        else:
            raise NotImplementedError

