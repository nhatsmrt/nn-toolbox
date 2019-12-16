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
from ...learner import SupervisedLearner


class SupervisedImageLearner(SupervisedLearner):
    def __init__(
            self, train_data: DataLoader, val_data: DataLoader, model: Module,
            criterion: Module, optimizer: Optimizer,
            mixup: bool=False, mixup_alpha: float=0.4, device=get_device()
    ):
        super(SupervisedImageLearner, self).__init__(
            train_data=train_data, val_data=val_data, model=model,
            criterion=criterion, optimizer=optimizer, mixup=mixup, mixup_alpha=mixup_alpha
        )
        self._device = device

    def learn_one_iter(self, images: Tensor, labels: Tensor):
        data = self._cb_handler.on_batch_begin({'inputs': images, 'labels': labels}, True)
        images = data['inputs']
        labels = data['labels']

        if self._mixup:
            images, labels = self._mixup_transformer.transform_data(images, labels)

        loss = self.compute_loss(images, labels, True)

        if self._cb_handler.on_backward_begin():
            loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()

            if self._device.type == 'cuda':
                mem = torch.cuda.memory_allocated(self._device)
                self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
            else:
                self._cb_handler.on_batch_end({"loss": loss})
            # self._cb_handler.on_batch_end({"loss": loss})

    # def compute_loss(self, images: Tensor, labels: Tensor, train: bool) -> Tensor:
    #     old_criterion = self._criterion
    #     if self._mixup:
    #         self._criterion = self._mixup_transformer.transform_loss(self._criterion, self._model.training)
    #     ret = super().compute_loss(images, labels, train)
    #     self._criterion = old_criterion
    #     return ret

        # outputs = self._cb_handler.after_outputs({"output": self._model(images)}, train)
        #
        # return self._cb_handler.after_losses({"loss": criterion(outputs["output"], labels)}, train)

