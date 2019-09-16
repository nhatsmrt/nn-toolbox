import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List


__all__ = ['LanguageModel', 'TransformerLM']


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
        self.hidden = None

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (seq_length, batch_size, input_dim)
        :return: (seq_length, batch_size, vocab_size)
        """
        output, hidden = self.encoder(self.embedding(input), self.hidden) # (seq_length, batch_size, output_dim)
        if isinstance(hidden, Tensor):
            self.hidden = hidden.detach()
        else:
            self.hidden = tuple((h.detach() for h in hidden))
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

    @torch.no_grad()
    def compute_prob(self, input: Tensor, from_ind: int=2) -> float:
        """
        Evaluating the probability of a sequence from a certain index

        :param input: a single sentence. (seq_len, )
        :param from_ind: the point to start evaluating probability
        :return: probability of the sentence
        """
        if from_ind < 0:
            return self.compute_prob(input, len(input) + from_ind)

        representation = self.encoder(self.embedding(input.unsqueeze(1)))[0]
        score = self.head(representation)[from_ind - 1:-1]  # (seq_len - from_ind + 1, batch_size, vocab_size)
        all_probs = torch.softmax(score, dim=-1)  # (seq_len - from_ind + 1, batch_size, vocab_size)
        final_prob = 1.0
        for ind in range(len(all_probs)):
            final_prob *= all_probs[ind, 0, input[ind + from_ind]]
            print(final_prob)
        return final_prob.item()

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

    def reset_hidden(self):
        self.hidden = None


class TransformerLM(LanguageModel):
    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (seq_length, batch_size, input_dim)
        :return: (seq_length, batch_size, vocab_size)
        """
        return self.head(self.encoder(self.embedding(input))) # (seq_length, batch_size, vocab_size)

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
        input = input.unsqueeze(1)

        for _ in range(n_token_gen):
            output = self.encoder(self.embedding(input))
            score = self.head(output[output.shape[0] - 1:]) # (1, batch_size, vocab_size)
            input = torch.max(F.softmax(score, dim=-1), -1)[1] # (1, batch_size)
            complete_sentence.append(input.cpu().detach().item())

        return complete_sentence

