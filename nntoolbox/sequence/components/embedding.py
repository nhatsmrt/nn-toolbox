from torch import nn, Tensor
import torch
import torch.nn.functional as F


__all__ = ['AdditiveContextEmbedding', 'TiedOutputEmbedding']


class AdditiveContextEmbedding(nn.Embedding):
    """
    The embedding weights are fixed, except for a context vector c shared by all embedding:

    embedding(x) = w_x + c
    """
    def __init__(
            self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
            norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None
    ):
        super(AdditiveContextEmbedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx,
            max_norm, norm_type, scale_grad_by_freq, sparse, _weight
        )
        self.weight.requires_grad = False
        self.context = nn.Parameter(torch.zeros(embedding_dim).to(self.weight.dtype))

    def forward(self, input):
        return super().forward(input) + self.context


class TiedOutputEmbedding(nn.Module):
    """
    Tie the weight of input embedding to the output layer

    References:

        Ofir Press and Lior Wolf. "Using the Output Embedding to Improve Language Models."
        https://arxiv.org/pdf/1608.05859.pdf
    """
    def __init__(self, emb: nn.Embedding, bias: bool=True):
        super().__init__()
        self.weight = nn.Parameter(emb.weight, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.weight.shape[0]), requires_grad=True) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

