import torch
from torch import nn, Tensor
from typing import Sequence, Callable, Optional


__all__ = [
    'LambdaLayer', 'ScalingLayer', 'ResidualLinearBlock',
    'LinearlyAugmentedFF', 'HighwayLayer', 'SquareUnitLinear',
    'QuadraticPolynomialLayer', 'MLP'
]


class LambdaLayer(nn.Module):
    """
    Implement a quick layer wrapper for a function

    Useful for stateless layer (e.g without parameters)
    """
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        super(LambdaLayer, self).__init__()
        self.fn = fn

    def forward(self, input: Tensor) -> Tensor:
        return self.fn(input)


class ScalingLayer(LambdaLayer):
    """
    References:

        Christian Szegedy et al. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning."
        https://arxiv.org/pdf/1602.07261.pdf
    """
    def __init__(self, scale: float=0.1):
        super(ScalingLayer, self).__init__(lambda inp: inp * scale)


class ResidualLinearBlock(nn.Module):
    """
    A two-layer linear block with residual connection:

    y = f(w_2f(w_1 x + b_1) + b_2) + x
    """
    def __init__(
            self, in_features: int, activation: Callable[..., nn.Module]=nn.ReLU,
            bias: bool=True, use_dropout: bool=False, drop_rate: float=0.5
    ):
        super(ResidualLinearBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features, bias=bias),
                activation(),
                nn.Dropout(drop_rate) if use_dropout else nn.Identity(),
                nn.Linear(in_features=in_features, out_features=in_features, bias=bias),
                activation()
            )
        )

    def forward(self, input):
        return input + self._modules["main"](input)


class LinearlyAugmentedFF(nn.Module):
    """
    Based on https://link.springer.com/chapter/10.1007/978-3-642-35289-8_13
    """
    def __init__(self, in_features: int, out_features: int, activation: Callable[..., nn.Module]=nn.Identity):
        super(LinearlyAugmentedFF, self).__init__()
        self._fc = nn.Linear(in_features, out_features)
        self._a = activation()

    def forward(self, x):
        op = self._fc(x) + torch.sum(x, dim=-1, keepdim=True)
        op = self._a(op)

        return op


class HighwayLayer(nn.Module):
    """
    Highway layer:

    y = T(x) * H(x) + (1 - T(x)) * x

    Reference:

    https://arxiv.org/pdf/1505.00387.pdf
    """
    def __init__(self, in_features: int, main: nn.Module, gate: Optional[nn.Module]=None):
        """
        :param in_features: Number of features of each input
        :param main: The main network H(x). Take input of with in_features and return output with in_features
        :param gate: the gating function. Take input of with in_features and return output with in_features
        """
        super(HighwayLayer, self).__init__()
        if gate is None:
            self._gate = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features),
                nn.Sigmoid()
            )
        else:
            self._gate = gate
        self._main = main

    def forward(self, input):
        """
        :param input: (batch_size, in_features)
        :return: output: (batch_size, in_features)
        """
        gate = self._gate(input)
        return gate * self._main(input) + (1 - gate) * input


class SquareUnitLinear(nn.Linear):
    """
    Augment input with square units:

    g(x) = W concat([x, x^2]) + b

    Reference:

    Flake, Gary. "Square Unit Augmented, Radially Extended, Multilayer Perceptrons." Neural Network: Tricks of the Trade
    """
    def __init__(self, in_features, out_features, bias: bool=True):
        super(SquareUnitLinear, self).__init__(in_features=in_features * 2, out_features=out_features, bias=bias)

    def forward(self, input):
        input = torch.cat([input, input * input], dim=-1)
        return super(SquareUnitLinear, self).forward(input)


class QuadraticPolynomialLayer(nn.Linear):
    """
    h(x) = sigma( sum_k(A_k x)^2 + bx + c)

    References:

        Bergstra et al. "Quadratic Polynomials Learn Better Image Features"
    """
    def __init__(self, in_features: int, out_features: int, threshold: float=0.0):
        super(QuadraticPolynomialLayer, self).__init__(in_features, out_features, False)
        self.linear_feature = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)

    def forward(self, input):
        weight = self.weight.t()
        quadratic_features = (weight.unsqueeze(0) * input.unsqueeze(-1)).pow(2).sum(1)
        return quadratic_features + self.linear_feature(input) + self.threshold


class MLP(nn.Sequential):
    """
    Implement a generic multilayer perceptron
    """
    def __init__(
            self, in_features: int, out_features: int, hidden_layer_sizes: Sequence[int]=(512,),
            activation: Callable[..., Tensor]=nn.ReLU, bn_final: bool=False,
            drop_ps=(0.5, 0.5), use_batch_norm: bool=True
    ):
        layers = []
        if isinstance(drop_ps, float):
            drop_ps = [drop_ps for _ in range(len(hidden_layer_sizes) + 1)]

        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                in_features = in_features
            else:
                in_features = hidden_layer_sizes[i - 1]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=in_features))
            drop_p = drop_ps[i]
            if drop_p != 0:
                layers.append(nn.Dropout(p=drop_p))
            layers.append(nn.Linear(
                in_features=in_features,
                out_features=hidden_layer_sizes[i]
            ))
            layers.append(activation())

        if bn_final and use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_layer_sizes[-1], momentum=0.001)) #follows fast ai
        if drop_ps[-1] != 0:
            layers.append(nn.Dropout(p=drop_ps[-1]))
        layers.append(nn.Linear(in_features=hidden_layer_sizes[-1], out_features=out_features))
        super(MLP, self).__init__(*layers)
