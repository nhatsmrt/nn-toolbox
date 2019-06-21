from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from ...utils import load_model, get_device
import torch
from typing import Iterable, Dict
from ...callbacks import CallbackHandler, Callback
from ...metrics import Metric
from ...transforms import MixupTransformer


class SupervisedImageLearner:

    def __init__(
            self, train_data: DataLoader, val_data: DataLoader, model: Module,
            criterion: Module, optimizer: Optimizer, mixup=False, mixup_alpha=0.4, device=get_device()
    ):
        self._train_data = train_data
        self._val_data = val_data
        self._model = model.to(device)
        self._criterion = criterion.to(device)
        self._optimizer = optimizer
        self._device = device
        self._mixup = mixup
        if mixup:
            self._mixup_transformer = MixupTransformer(alpha=mixup_alpha)

    def learn(
            self,
            n_epoch: int, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, final_metric: str='accuracy', load_path=None
    ) -> float:
        if load_path is not None:
            load_model(self._model, load_path)

        self._cb_handler = CallbackHandler(callbacks, metrics, final_metric)
        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._model.train()

            for images, labels in self._train_data:
                self.learn_one_iter(images, labels)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        return self._cb_handler.on_train_end()

    def learn_one_iter(self, images, labels):
        images = images.to(self._device)
        labels = labels.to(self._device)

        if self._mixup:
            images, labels = self._mixup_transformer.transform_data(images, labels)

        self._optimizer.zero_grad()
        loss = self.compute_loss(images, labels)
        loss.backward()
        self._optimizer.step()
        if self._device.type == 'cuda':
            mem = torch.cuda.memory_allocated(self._device)
            self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
        else:
            self._cb_handler.on_batch_end({"loss": loss.cpu()})

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
        if self._mixup:
            criterion = self._mixup_transformer.transform_loss(self._criterion, self._model.training)
        else:
            criterion = self._criterion

        return criterion(self._model(images), labels)

