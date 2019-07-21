from torch import Tensor
from nntoolbox.learner import Learner
from ...callbacks import Callback, CallbackHandler
from ...metrics import Metric
from typing import Tuple, List, Dict


__all__ = ['SiameseLearner']


class SiameseLearner(Learner):
    """
    Abstraction for training of neural network in siamese style
    """

    def learn(self, n_epoch, callbacks: List[Callback], metrics: Dict[str, Metric], final_metric: str) -> float:
        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()

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
        images_1, labels_1, images_2, labels_2 = data[0][0], data[0][1], data[1][0], data[1][1]
        first = self._cb_handler.on_batch_begin({"inputs": images_1, "labels": labels_1}, True)
        images_1, labels_1 = first["inputs"], first["labels"]
        second = self._cb_handler.on_batch_begin({"inputs": images_2, "labels": labels_2}, True)
        images_2, labels_2 = second["inputs"], second["labels"]

        embeddings_1, embeddings_2 = self.compute_embeddings(images_1, images_2, True)
        labels = labels_1 == labels_2
        loss = self.compute_loss(embeddings_1, embeddings_2, labels, True)
        print(loss)
        loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()

            self._cb_handler.on_batch_end({"loss": loss})

    def compute_embeddings(self, images_1, images_2, train: bool) -> Tuple[Tensor, Tensor]:
        embedding_1, embedding_2 = self._model(images_1), self._model(images_2)
        return embedding_1, embedding_2

    def compute_loss(self, embeddings_1, embeddings_2, labels, train: bool) -> Tensor:
        return self._cb_handler.after_losses(
            {"loss": self._criterion(embeddings_1, embeddings_2, labels)}, train
        )["loss"]

    def evaluate(self) -> bool:
        return False