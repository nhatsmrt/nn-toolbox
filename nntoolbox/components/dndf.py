"""Deep Neural Decision Forest"""
from functools import partial
import torch
from torch import nn, Tensor
from .components import ModifyByLambda
from .merge import Multiply, Mean


__all__ = ['DNDFTree', 'DNDF']


class DNDFTree(nn.Module):
    """
    Based on Deep Neural Decision Forest, but with the leaf node parameterized for end-to-end training, and the decision
    trees balanced.

    References:

        Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, Samuel Rota Bulo. "Deep Neural Decision Forests."
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf
    """
    def __init__(
            self, in_features: int, out_features: int, tree_depth: int,
            output_activation=partial(nn.Softmax, dim=1)
    ):
        assert tree_depth > 1 and in_features > 1
        super().__init__()
        n_leaves = 2 ** tree_depth

        self.leaves = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_features, out_features), output_activation(dim=1)) for _ in range(n_leaves)]
        )
        decision_functions = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid()) for _ in range(n_leaves - 1)]
        )

        routings = []
        for l in range(n_leaves):
            route = []

            node_ind = 0
            index = l
            for d in range(tree_depth):
                if index % 2 == 0:
                    route.append(decision_functions[node_ind])
                else:
                    route.append(ModifyByLambda(decision_functions[node_ind], lambda t: 1.0 - t))
                node_ind = node_ind * 2 + 1 + index % 2
                index = index >> 1
            routings.append(Multiply(route))
        self.routings = nn.ModuleList(routings)

    def forward(self, input: Tensor) -> Tensor:
        routings = [route(input) for route in self.routings]
        leaves = [leaf(input) for leaf in self.leaves]
        return (torch.stack(routings, dim=-1) * torch.stack(leaves, dim=-1)).sum(dim=-1)


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
