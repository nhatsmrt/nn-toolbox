from torch import nn, Tensor
import torch
import torch.nn.functional as F
from ...utils import dropout_mask


__all__ = ['AdditiveContextEmbedding', 'TiedOutputEmbedding', 'EmbeddingDropout', 'SinusoidPositionalEncoding']


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


class EmbeddingDropout(nn.Module):
    """
    "Zero out" the same word across time step for each batch. E.g:

    "The cat hates the dog." -> "- cat hates - dog."

    Based on fastai's notebook implementation (which is slightly slower than as suggested in the paper, but easier
    to implement).

    References:

        FastAI Course 2 V3 Notebook:
        https://github.com/fastai/course-v3/blob/master/nbs/dl2/12a_awd_lstm.ipynb

        Yarin Gal and Zoubin Ghahramani. "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks."
        https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf

        Stephen Merity, Nitish Shirish Keskar, Richard Socher. "Regularizing and Optimizing LSTM Language Models."
        https://arxiv.org/abs/1708.02182
    """
    def __init__(self, emb: nn.Embedding, drop_p: float=0.5):
        super().__init__()
        self.drop_p = drop_p
        self.emb = emb

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            mask = dropout_mask(self.emb.weight, (self.emb.weight.shape[0], 1), self.drop_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(
            input, masked_embed, padding_idx=self.emb.padding_idx,
            max_norm=self.emb.max_norm, norm_type=self.emb.norm_type,
            scale_grad_by_freq=self.emb.scale_grad_by_freq, sparse=self.emb.sparse
        )


class SinusoidPositionalEncoding(nn.Module):
    """
    Sinusoid Positional Encoding for Transformers. (UNTESTED)

    References:

        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
        Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
        "Attention Is All You Need." https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    """
    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (seq_len, batch_size, n_features)
        :return: same shape
        """
        pos_encoding = torch.arange(
            0, input.shape[0], dtype=input.dtype, device=input.device
        )[:, None, None, None].repeat((1, input.shape[1], input.shape[2] // 2, 1))
        dim_encoding = torch.arange(
            0, input.shape[2], 2, dtype=input.dtype, device=input.device
        )[None, None, :, None].repeat((input.shape[0], input.shape[1], 1, 1))

        encoding_even = torch.sin(pos_encoding / torch.pow(10000, dim_encoding / input.shape[2]))
        encoding_odd = torch.cos(pos_encoding / torch.pow(10000, dim_encoding / input.shape[2]))
        encoding = torch.cat([encoding_even, encoding_odd], dim=-1).view(input.shape[0], input.shape[1], input.shape[2])
        return input + encoding
