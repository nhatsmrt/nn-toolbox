from torch import nn
from torchtext.data import Iterator
from nntoolbox.utils import get_device
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from torch.optim import Optimizer


# UNTESTED
class SequenceClassifierLearner:
    def __init__(
            self, train_iterator: Iterator, val_iterator: Iterator, model: nn.Module,
            criterion: nn.Module, optimizer: Optimizer, device=get_device()
    ):
        self._train_iterator = train_iterator
        self._val_iterator = val_iterator
        self._model = model.to(device)
        self._optimizer = optimizer
        self._criterion = criterion.to(device)
        self._device = device

    def learn(self, n_epoch, callbacks, metrics, final_metric):
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
        texts = texts.to(self._device)
        text_lengths = text_lengths.to(self._device)

        self._optimizer.zero_grad()
        loss = self.compute_loss(texts, text_lengths, batch.label)
        loss.backward()
        self._optimizer.step()
        # del texts, text_lengths
        # torch.cuda.empty_cache()
        # if self._device.type == 'cuda':
        #     mem = torch.cuda.memory_allocated(self._device)
        #     self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
        # else:
        #     self._cb_handler.on_batch_end({"loss": loss.cpu()})
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
            texts = texts.to(self._device)
            text_lengths = text_lengths.to(self._device)

            outputs = self._model(texts, text_lengths)
            all_outputs.append(outputs)
            all_labels.append(batch.label.unsqueeze(-1))
            loss += float(self.compute_loss(texts, text_lengths, batch.label)) * len(outputs)
            total_data += len(outputs)

        loss /= total_data
        logs = dict()
        logs["loss"] = loss
        logs["outputs"] = torch.cat(all_outputs, dim=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)

    def compute_loss(self, texts, text_lengths, labels):
        outputs = self._model(texts, text_lengths)
        return self._criterion(outputs, labels.long())
