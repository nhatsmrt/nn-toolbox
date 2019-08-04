from ...callbacks import Callback, CallbackHandler
from ...metrics import Metric
from ..models import LanguageModel
from ...utils import grab_next_batch
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from typing import List, Optional
from torchtext.data import Iterator


__all__ = ['LanguageModelLearner']


class LanguageModelLearner:
    """
    Train a language model (predicting the next word given the previous word)
    """
    def __init__(
            self, train_iterator: Iterator, val_iterator: Iterator,
            model: LanguageModel, optimizer: Optimizer, criterion: nn.Module
    ):
        self._train_iterator = train_iterator
        self._val_iterator = val_iterator
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion

    def learn(self, n_epoch: int, callbacks: Optional[List[Callback]]=None, metrics: Optional[List[Metric]]=None):
        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics)
        self._cb_handler.on_train_begin()

        for e in range(n_epoch):
            self._cb_handler.on_epoch_begin()
            self._model.train()
            for example in self._train_iterator:
                self.learn_one_iter(example)

            stop_training = self.evaluate()
            if stop_training:
                break

        return self._cb_handler.on_train_end()

    def learn_one_iter(self, example):
        inputs = self._cb_handler.on_batch_begin({'text': example.text, 'target': example.target}, True)
        text, target = inputs['text'], inputs['target']
        output = self.compute_output(text, True)
        loss = self.compute_loss(output, target, True)

        loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()

            self._cb_handler.on_batch_end({'loss': loss.cpu().detach()})

    @torch.no_grad()
    def evaluate(self) -> bool:
        self._model.eval()
        example = grab_next_batch(self._val_iterator)
        inputs = self._cb_handler.on_batch_begin({'text': example.text, 'target': example.target}, True)
        text, target = inputs['text'], inputs['target']
        output = self.compute_output(text, False)
        loss = self.compute_loss(output, target, False).cpu().detach().item()
        return self._cb_handler.on_epoch_end({
            "loss": loss,
            "outputs": output.cpu().detach(),
            "labels": target.cpu()
        })


        # all_outputs = []
        # all_labels = []
        # total_data = 0
        # loss = 0

        # for example in self._val_iterator:
        #     inputs = self._cb_handler.on_batch_begin({'text': example.text, 'target': example.target}, True)
        #     text, target = inputs['text'], inputs['target']
        #     output = self.compute_output(text, False)
        #
        #     all_outputs.append(output.cpu().detach())
        #     all_labels.append(target.cpu())
        #     loss += self.compute_loss(output, target, False).cpu().detach().item() * text.shape[1]
        #     total_data += text.shape[1]
        #
        # loss /= total_data
        # logs = dict()
        # logs["loss"] = loss
        # logs["outputs"] = torch.cat(all_outputs, dim=1)
        # logs["labels"] = torch.cat(all_labels, dim=0)
        #
        # return self._cb_handler.on_epoch_end(logs)

    def compute_output(self, text: Tensor, train: bool) -> Tensor:
        return self._cb_handler.after_outputs({'output': self._model(text).permute(0, 2, 1)}, train)['output']

    def compute_loss(self, output: Tensor, target: Tensor, train: bool) -> Tensor:
        return self._cb_handler.after_losses({'loss': self._criterion(output, target)}, train)['loss']
