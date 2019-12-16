import torch
from torch.nn import Module
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .callbacks import Callback, CallbackHandler
from .metrics import Metric
from .utils import get_device, load_model
from .transforms import MixupTransformer
from typing import Iterable, Dict


__all__ = ['Learner', 'SupervisedLearner', 'DistillationLearner']


class Learner:
    def __init__(
            self, train_data: DataLoader, val_data: DataLoader,
            model: Module, criterion: Module, optimizer: Optimizer
    ):
        self._train_data, self._val_data = train_data, val_data
        self._model, self._criterion, self._optimizer = model, criterion, optimizer

        
class SupervisedLearner(Learner):
    def __init__(
            self, train_data: DataLoader, val_data: DataLoader,  model: Module, 
            criterion: Module, optimizer: Optimizer, device=get_device(), mixup: bool=False, mixup_alpha: float=0.4
    ):
        super().__init__(train_data, val_data, model, criterion, optimizer)
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

        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()

        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._model.train()
            self._cb_handler.on_epoch_begin()

            for inputs, labels in self._train_data:
                self.learn_one_iter(inputs, labels)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        return self._cb_handler.on_train_end()

    def learn_one_iter(self, inputs: Tensor, labels: Tensor):
        data = self._cb_handler.on_batch_begin({'inputs': inputs, 'labels': labels}, True)
        inputs = data['inputs']
        labels = data['labels']

        if self._mixup:
            inputs, labels = self._mixup_transformer.transform_data(inputs, labels)

        self._optimizer.zero_grad()
        loss = self.compute_loss(inputs, labels, True)
        loss.backward()
        self._optimizer.step()
        if self._device.type == 'cuda':
            mem = torch.cuda.memory_allocated(self._device)
            self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
        else:
            self._cb_handler.on_batch_end({"loss": loss})

    @torch.no_grad()
    def evaluate(self) -> float:
        self._model.eval()
        all_outputs = []
        all_labels = []
        total_data = 0
        loss = 0

        for inputs, labels in self._val_data:
            data = self._cb_handler.on_batch_begin({'inputs': inputs, 'labels': labels}, False)
            inputs = data['inputs']
            labels = data['labels']

            all_outputs.append(self.compute_outputs(inputs, False))
            all_labels.append(labels)
            loss += self.compute_loss(inputs, labels, False).cpu().item() * len(inputs)
            total_data += len(inputs)

        loss /= total_data

        logs = dict()
        logs["loss"] = loss
        logs["outputs"] = torch.cat(all_outputs, dim=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)

    def compute_outputs(self, inputs: Tensor, train: bool) -> Tensor:
        return self._cb_handler.after_outputs({"output": self._model(inputs)}, train)["output"]

    def compute_loss(self, inputs: Tensor, labels: Tensor, train: bool) -> Tensor:
        if self._mixup:
            criterion = self._mixup_transformer.transform_loss(self._criterion, self._model.training)
        else:
            criterion = self._criterion
        outputs = self.compute_outputs(inputs, train)

        return self._cb_handler.after_losses({"loss": criterion(outputs, labels)}, train)["loss"]


class DistillationLearner(SupervisedLearner):
    """
    Distilling Knowledge from a big teacher network to a smaller model (UNTESTED)

    References:

        Geoffrey Hinton, Oriol Vinyals, Jeff Dean. "Distilling the Knowledge in a Neural Network."
        https://arxiv.org/abs/1503.02531

        TTIC Distinguished Lecture Series - Geoffrey Hinton.
        https://www.youtube.com/watch?v=EK61htlw8hY
    """
    def __init__(
            self, train_data: DataLoader, val_data: DataLoader,
            model: Module, teacher: Module, criterion: Module, optimizer: Optimizer,
            temperature: float, teacher_weight: float, hard_label_weight: float, device = get_device()
    ):
        assert temperature >= 1.0 and teacher_weight >= 1.0 and hard_label_weight > 1.0
        super().__init__(train_data, val_data, model, criterion, optimizer, device)
        self._teacher = teacher.to(device)
        self.temperature, self.teacher_weight, self.hard_label_weight = temperature, teacher_weight, hard_label_weight

    def compute_loss(self, inputs: Tensor, labels: Tensor) -> Tensor:
        model_outputs = self._model(inputs)
        hard_label_loss = self._criterion(model_outputs, labels)
        teacher_outputs = self._teacher(inputs)
        soft_label_loss = -(teacher_outputs * torch.log_softmax(model_outputs / self.temperature, dim=1)).sum(1)
        return hard_label_loss * self.hard_label_weight + self.teacher_weight * soft_label_loss

