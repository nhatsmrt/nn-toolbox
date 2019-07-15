import torch
from torch import nn
from torchtext.data import Iterator
from nntoolbox.utils import get_device
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from torch.optim import Optimizer
from typing import Optional, List


# UNTESTED
class SequenceClassifierLearner:
    def __init__(
            self, train_iterator: Iterator, val_iterator: Iterator, model: nn.Module,
            criterion: nn.Module, optimizer: Optimizer, device=get_device()
    ):
        self._train_data = self._train_iterator = train_iterator
        self._val_iterator = val_iterator
        self._model = model.to(device)
        self._optimizer = optimizer
        self._criterion = criterion.to(device)
        self._device = device

    def learn(
            self, n_epoch: int, callbacks: Optional[List[Callback]]=None,
            metrics: Optional[List[Metric]]=None, final_metric: str='accuracy'
    ) -> float:
        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)

        for e in range(n_epoch):
            self._model.train()
            for batch in self._train_iterator:
                self.learn_one_iter(batch)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        return self._cb_handler.on_train_end()

    def learn_one_iter(self, batch):
        texts, text_lengths = batch.text
        # texts = texts.to(self._device)
        # text_lengths = text_lengths.to(self._device)
        data = self._cb_handler.on_batch_begin(
            {"texts": texts, "text_lengths": text_lengths, "labels": batch.label}, True
        )
        texts, text_lengths, labels = data["texts"], data["text_lengths"], data["labels"]

        self._optimizer.zero_grad()
        loss = self.compute_loss(texts, text_lengths, labels, True)
        loss.backward()
        self._optimizer.step()
        self._cb_handler.on_batch_end({"loss": loss})

    @torch.no_grad()
    def evaluate(self):
        self._model.eval()
        all_outputs = []
        all_labels = []
        total_data = 0
        loss = 0

        for batch in self._val_iterator:
            texts, text_lengths = batch.text
            # texts = texts.to(self._device)
            # text_lengths = text_lengths.to(self._device)
            data = self._cb_handler.on_batch_begin(
                {"texts": texts, "text_lengths": text_lengths, "labels": batch.label}, True
            )
            texts, text_lengths, labels = data["texts"], data["text_lengths"], data["labels"]

            outputs = self._model(texts, text_lengths)
            all_outputs.append(outputs)
            all_labels.append(labels.unsqueeze(-1))
            loss += float(self.compute_loss(texts, text_lengths, labels, False)) * len(outputs)
            total_data += len(outputs)

        loss /= total_data
        logs = dict()
        logs["loss"] = loss
        logs["outputs"] = torch.cat(all_outputs, dim=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)

    def compute_loss(self, texts, text_lengths, labels, train: bool):
        output = self._cb_handler.after_outputs({"output": self._model(texts, text_lengths)}, train)["output"]
        return self._cb_handler.after_losses({"loss": self._criterion(output, labels.long())}, train)["loss"]
