from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from ...utils import load_model, get_device
import torch
from typing import Iterable, Dict
from ...callbacks import CallbackHandler, Callback
from ...metrics import Metric
from ...transforms import MixupTransformer


class SupervisedImageLearner:

    def __init__(
            self, train_data: DataLoader, val_data: DataLoader, model: Module,
            criterion: Module, optimizer: Optimizer,
            mixup: bool=False, mixup_alpha: float=0.4, device=get_device()
    ):
        self._train_data = train_data
        self._val_data = val_data
        # self._model = model.to(device)
        self._model = model
        # self._criterion = criterion.to(device)
        self._criterion = criterion
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

        self._cb_handler = CallbackHandler(self, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()
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

    def learn_one_iter(self, images: Tensor, labels: Tensor):
        # images = images.to(self._device)
        # labels = labels.to(self._device)
        data = self._cb_handler.on_batch_begin({'inputs': images, 'labels': labels}, True)
        images = data['inputs']
        labels = data['labels']

        if self._mixup:
            images, labels = self._mixup_transformer.transform_data(images, labels)

        loss = self.compute_loss(images, labels)

        loss.backward()
        self._cb_handler.after_backward()

        self._optimizer.step()
        if self._cb_handler.after_step():
            self._optimizer.zero_grad()

        if self._device.type == 'cuda':
            mem = torch.cuda.memory_allocated(self._device)
            self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
        else:
            self._cb_handler.on_batch_end({"loss": loss})
        # self._cb_handler.on_batch_end({"loss": loss})

    @torch.no_grad()
    def evaluate(self) -> float:
        self._model.eval()
        all_outputs = []
        all_labels = []
        total_data = 0
        loss = 0

        for images, labels in self._val_data:
            data = self._cb_handler.on_batch_begin({"inputs": images, "labels": labels}, False)
            images, labels = data["inputs"], data["labels"]

            # all_outputs.append(self._model(images.to(self._device)))
            all_outputs.append(self._model(images))
            all_labels.append(labels.cpu())
            # loss += self.compute_loss(images.to(self._device), labels.to(self._device)).cpu().item() * len(images)
            loss += self.compute_loss(images, labels).cpu().item() * len(images)
            total_data += len(images)

        loss /= total_data

        logs = dict()
        logs["loss"] = loss
        logs["outputs"] = torch.cat(all_outputs, dim=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)

    def compute_loss(self, images: Tensor, labels: Tensor) -> Tensor:
        if self._mixup:
            criterion = self._mixup_transformer.transform_loss(self._criterion, self._model.training)
        else:
            criterion = self._criterion

        outputs = self._cb_handler.after_outputs({"output": self._model(images)}, True)

        return criterion(outputs["output"], labels)

