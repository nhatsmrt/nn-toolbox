import torch
from torch import Tensor
from nntoolbox.learner import Learner
from ...callbacks import Callback, CallbackHandler
from ...metrics import Metric
from ..utils import PairSelector
from typing import Tuple, List, Dict


__all__ = ['SiameseLearner']


class SiameseLearner(Learner):
    """
    Abstraction for training of neural network in siamese style
    """

    def learn(
            self, n_epoch, selector: PairSelector, callbacks: List[Callback],
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
        # return self._cb_handler.on_train_end()

    def learn_one_iter(self, data):
        images, labels = data
        data = self._cb_handler.on_batch_begin({"inputs": images, "labels": labels}, True)
        images, labels = data["inputs"], data["labels"]
        embeddings = self._cb_handler.after_outputs({"embeddings": self._model(images)}, True)["embeddings"]
        embeddings_1, embeddings_2, labels = self._selector.return_pairs(embeddings, labels)

        loss = self.compute_loss(embeddings_1, embeddings_2, labels, True)
        print(loss)
        loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()

            self._cb_handler.on_batch_end({"loss": loss})

    def compute_loss(self, embeddings_1, embeddings_2, labels, train: bool) -> Tensor:
        return self._cb_handler.after_losses(
            {"loss": self._criterion(embeddings_1, embeddings_2, labels)}, train
        )["loss"]

    def evaluate(self) -> bool:
        return False