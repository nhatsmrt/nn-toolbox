import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List


__all__ = ['LanguageModel']


class LanguageModel(nn.Module):
    """
    General Language Model (UNTESTED):

    p(x_t|x_{<t}) = prod_{t' < t} p(x_{t'} | x_{<t'})
    """
    def __init__(self, embedding: nn.Embedding, encoder: nn.Module, head: nn.Module):
        """
        :param encoder: encoding the hidden states of words up to now
        :param head: compute score for each word given the input
        """
        super(LanguageModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.head = head

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (seq_length, batch_size, input_dim)
        :return: (seq_length, batch_size, vocab_size)
        """
        output = self.encoder(self.embedding(input))[0] # (seq_length, batch_size, output_dim)
        return self.head(output) # (seq_length, batch_size, vocab_size)

    def get_encoder(self) -> nn.Module:
        """
        Extract the encoder for downstream task

        :return: encoder
        """
        return self.encoder

    def get_embedding(self) -> nn.Module:
        """
        Extract the embedding for downstream task

        :return: encoder
        """
        return self.embedding

    def replace_head(self, head: nn.Module):
        """
        Replace the head (e.g for fine-tuning on another domain)

        :param head:
        """
        self.head = head

    def compute_prob(self, input: Tensor) -> Tensor:
        """
        :param input: a single sentence. (seq_len, )
        :return: probability of the sentence
        """
        representation = self.encoder(self.embedding(input.unsqueeze(1)))[0]
        score = self.head(representation) # (seq_len, batch_size, vocab_size)
        return F.softmax(score, dim=-1).prod()

    @torch.no_grad()
    def complete(self, input: Tensor, n_token_gen: int=1) -> List[int]:
        """
        Complete the sentence

        :param input: a single partial sentence. (seq_len, )
        :param n_token_gen: number of tokens to generated
        :return: the complete sentence: (seq_len + n_token_gen, input_dim)
        """
        self.embedding.eval()
        self.encoder.eval()
        self.head.eval()

        complete_sentence = [token.item() for token in input.view(-1).cpu().detach()]
        input, states = input.unsqueeze(1), None

        for _ in range(n_token_gen):
            output, states = self.encoder(self.embedding(input), states)
            score = self.head(output[output.shape[0] - 1:]) # (1, batch_size, vocab_size)
            input = torch.max(F.softmax(score, dim=-1), -1)[1] # (1, batch_size)
            complete_sentence.append(input.cpu().detach().item())

        return complete_sentence
