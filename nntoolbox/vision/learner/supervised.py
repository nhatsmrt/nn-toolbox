from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ...utils import load_model, get_device
import torch
from typing import Iterable, Dict
from ...callbacks import CallbackHandler, Callback
from ...metrics import Metric


class SupervisedImageLearner:

    def __init__(
            self, train_data: DataLoader, val_data: DataLoader, model: Module, criterion: Module,
            optimizer: Optimizer, val_metric=None, use_scheduler=False, device=get_device()
    ):
        self._train_data = train_data
        self._val_data = val_data
        self._model = model.to(device)
        self._criterion = criterion.to(device)
        self._optimizer = optimizer
        self._val_metric = val_metric
        self._device = device

        if use_scheduler:
            self._lr_scheduler = ReduceLROnPlateau(self._optimizer, mode='max')
        else:
            self._lr_scheduler = None

    def learn(
            self,
            n_epoch: int, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, load_path=None
    ) -> float:
        if load_path is not None:
            load_model(self._model, load_path)

        self._cb_handler = CallbackHandler(callbacks, metrics)
        val_metrics = []
        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._model.train()

            for images, labels in self._train_data:
                # images, labels = self._cb_handler.on_batch_begin(images, labels)
                self.learn_one_iter(images, labels)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        return max(val_metrics) if self._val_metric == 'accuracy' else min(val_metrics)

    def learn_one_iter(self, images, labels):
        self._optimizer.zero_grad()
        loss = self.compute_loss(images.to(self._device), labels.to(self._device))
        loss.backward()
        self._optimizer.step()
        self._cb_handler.on_batch_end({"loss": loss})

    @torch.no_grad()
    def evaluate(self) -> float:
        self._model.eval()
        all_outputs = []
        all_labels = []
        total_data = 0
        loss = 0

        for images, labels in self._val_data:
            all_outputs.append(self._model(images.to(self._device)))
            all_labels.append(labels)
            loss += self.compute_loss(images.to(self._device), labels.to(self._device)).cpu().item() * len(images)
            total_data += len(images)

        loss /= total_data

        logs = dict()
        logs["loss"] = loss
        logs["outputs"] = torch.cat(all_outputs, dim=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)

    def compute_loss(self, images, labels) -> torch.Tensor:
        return self._criterion(self._model(images), labels)
