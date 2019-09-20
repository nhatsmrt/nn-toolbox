import torch
from torch import nn, Tensor
from typing import Optional
from ...components import MLP
import numpy as np
from ...init import sqrt_uniform_init


__all__ = ['CollabFiltering', 'NonLinearCF']


class CollabFiltering(nn.Module):
    """A simple collaborative model"""
    def __init__(self, n_users: int, n_products: int, embedding_dim: int):
        super().__init__()
        self.users = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.products = nn.Embedding(num_embeddings=n_products, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        sqrt_uniform_init(self)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        :param inputs: the pair of user-product of shape (batch_size, 2)
        :return: (batch_size, 1)
        """
        return (self.users(inputs[:, 0]) * self.products(inputs[:, 1])).sum(-1, keepdim=True)

    def get_score(self, users: Tensor, products: Tensor) -> Tensor:
        """
        Return the score of corresponding pairs of users-products

        :param users: (batch_size, )
        :param products: (batch_size, )
        """
        return self.forward(torch.stack((users, products), dim=-1))


class NonLinearCF(nn.Module):
    """
    A non-linear collaborative model. If no body model is provided, default to a one-hidden layer net
    """
    def __init__(
            self, n_users: int, n_products: int, user_dim: int, product_dim: int, body: Optional[nn.Module]=None
    ):
        super().__init__()
        self.users = nn.Embedding(num_embeddings=n_users, embedding_dim=user_dim)
        self.products = nn.Embedding(num_embeddings=n_products, embedding_dim=product_dim)
        sqrt_uniform_init(self)
        if body is None:
            self.body = MLP(
                in_features=product_dim + user_dim,
                hidden_layer_sizes=(2 ** int(np.log2(np.sqrt(product_dim + user_dim))), ),
                out_features=1
            )
        else:
            self.body = body

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Return the score of corresponding pairs of products and users

        :param inputs: the pair of user-product of shape (batch_size, 2)
        :return: (batch_size, 1)
        """
        features = torch.cat((self.users(inputs[:, 0]), self.products(inputs[:, 1])), dim=-1)
        return self.body(features)

    def get_score(self, users: Tensor, products: Tensor) -> Tensor:
        """
        Return the score of corresponding pairs of users-products

        :param users: (batch_size, )
        :param products: (batch_size, )
        """
        return self.forward(torch.stack((users, products), dim=-1))
