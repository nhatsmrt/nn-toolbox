from torch import nn
import torch


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
