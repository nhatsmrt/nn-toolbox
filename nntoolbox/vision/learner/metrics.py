import torch
from torch import Tensor
from nntoolbox.learner import Learner
from ...callbacks import Callback, CallbackHandler
from ...metrics import Metric
from ..utils import Selector
from typing import List, Dict, Tuple
from ..models import KNNClassifier
import numpy as np


__all__ = ['MetricLearner']


class MetricLearner(Learner):
    def learn(
            self, n_epoch, selector: Selector, callbacks: List[Callback],
            metrics: Dict[str, Metric], final_metric: str
    ) -> float:
        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()
        self._selector = selector

        for e in range(n_epoch):
            self._model.train()

            for data in self._train_data:
                self.learn_one_iter(data)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        print("Finish training")
        return 0.0

    def learn_one_iter(self, data: Tuple[Tensor, Tensor]):
        images, labels = data
        data = self._cb_handler.on_batch_begin({"inputs": images, "labels": labels}, True)
        images, labels = data["inputs"], data["labels"]
        embeddings = self.compute_embeddings(images, True)
        try:
            selected = self._selector.select(embeddings, labels)
        except:
            return

        loss = self.compute_loss(selected, True)
        # print(loss)
        loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()

            self._cb_handler.on_batch_end({"loss": loss})

    def compute_embeddings(self, images: Tensor, train: bool) -> Tensor:
        return self._cb_handler.after_outputs({"embeddings": self._model(images)}, train)["embeddings"]

    def compute_loss(self, selected: Tuple[Tensor, ...], train: bool) -> Tensor:
        return self._cb_handler.after_losses(
            {"loss": self._criterion(selected)}, train
        )["loss"]

    def evaluate(self) -> bool:
        self._model.eval()
        classifier = KNNClassifier(model=self._model, database=self._train_data)

        all_outputs = []
        all_labels = []
        all_bests = []

        total_data = 0
        loss = 0

        for data in self._val_data:
            images, labels = data
            data = self._cb_handler.on_batch_begin({"inputs": images, "labels": labels}, True)
            predictions, best, prediction_probs = classifier.predict(images)

            images, labels = data["inputs"], data["labels"]
            embeddings = self.compute_embeddings(images, True)
            try:
                selected = self._selector.select(embeddings, labels)
            except:
                continue

            loss = self.compute_loss(selected, True)

            all_outputs.append(predictions)
            all_bests.append(best)
            all_labels.append(labels.cpu())

            loss += self.compute_loss(selected, False).cpu().item() * len(images)
            total_data += len(images)
            break

        loss /= total_data

        logs = dict()

        logs["loss"] = loss
        logs["best"] = np.concatenate(all_bests, axis=0)
        logs["outputs"] = np.concatenate(all_outputs, axis=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)
