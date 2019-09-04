"""Deep Neural Decision Forest"""
from functools import partial
import torch
from torch import nn, Tensor
from .merge import Mean


__all__ = ['DNDFTree', 'DNDF']


class DNDFTree(nn.Module):
    """
    Based on Deep Neural Decision Forest, but with the leaf node parameterized for end-to-end training,
    and the decision trees balanced. Use BFS + DP for fast path computations

    References:

        Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, Samuel Rota Bulo. "Deep Neural Decision Forests."
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf
    """
    def __init__(
            self, in_features: int, out_features: int, tree_depth: int,
            output_activation=partial(nn.Softmax, dim=1)
    ):
        super().__init__()
        self.n_leaves = 2 ** tree_depth
        self.tree_depth = tree_depth
        self.out_features = out_features
        self.output_activation = output_activation()
        self.transform = nn.Linear(in_features, out_features * self.n_leaves + self.n_leaves - 1)

    def forward(self, input: Tensor) -> Tensor:
        features = self.transform(input)
        decision_nodes, leaves = torch.sigmoid(features[:, :self.n_leaves - 1]), \
                                 features[:, :self.out_features * self.n_leaves]
        neg_decision_nodes = 1.0 - decision_nodes
        leaves = self.output_activation(leaves.view(-1, self.out_features, self.n_leaves)).permute(2, 0, 1)
        routings = [1.0]

        for d in range(self.tree_depth):
            new_level = []
            for f in range(2 ** d - 1, 2 ** (d + 1) - 1):
                new_level.append(routings[f - 2 ** d + 1] * decision_nodes[:, f:f + 1])
                new_level.append(routings[f - 2 ** d + 1] * neg_decision_nodes[:, f:f + 1])
            routings = new_level

        return torch.stack([routings[i] * leaves[i] for i in range(self.n_leaves)], dim=-1).sum(-1)


class DNDF(Mean):
    """
    References:

        Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, Samuel Rota Bulo. "Deep Neural Decision Forests."
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf
    """
    def __init__(
            self, in_features: int, out_features: int, n_trees: int,
            tree_depth: int, output_activation=partial(nn.Softmax, dim=1)
    ):
        super().__init__([DNDFTree(in_features, out_features, tree_depth, output_activation) for _ in range(n_trees)])
