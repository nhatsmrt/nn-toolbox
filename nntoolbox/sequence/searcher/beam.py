import torch
from torch import Tensor
from ..models import LanguageModel
import numpy as np


__all__ = ['LMBeamSearcher']


class LMBeamSearcher:
    """
    Using beam search to iteratively add new word to the sentence.

    At each iteration, keep track of top K best sentences.

    (UNTESTED)
    """

    def __init__(self, model: LanguageModel, vocab_size: int, k: int=5):
        self.model = model
        self.k = k
        self.vocab_size = vocab_size

    def search(self, input: Tensor, n_iter: int) -> Tensor:
        current = [input]
        current_scores = [self.model.compute_prob(input)]

        for _ in range(n_iter):
            candidates = []
            candidates_scores = []

            for sequence in current:
                for new_word in range(self.vocab_size):
                    new_sequence = torch.cat(
                        [
                            sequence,
                            (torch.ones(1) * new_word).to(sequence.dtype).to(sequence.device)
                        ],
                        dim=0
                    )
                    candidates.append(new_sequence)
                    candidates_scores.append(self.model.compute_prob(new_sequence))

            top_ind = np.array(current_scores + candidates_scores).argsort()[-self.k:][::-1]
            current = (current + candidates)[top_ind]

        return current[np.argmax(current_scores)]
