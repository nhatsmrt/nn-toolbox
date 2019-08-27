"""Shunting Inhibition Modules"""
from torch import nn, Tensor, clamp
from .components import BiasLayer


__all__ = ['GeneralizedShuntingModule', 'GeneralizedShuntingMLP']


class GeneralizedShuntingModule(nn.Module):
    """
    Implement a module that exhibits the shunting inhibition mechanism:

    y = f(x) / (a + g(x))

    Difference from original implementation: clamping denominator.

    References:

        Ganesh Arulampalam, Abdesselam Bouzerdoum.
        "A generalized feedforward neural network architecture for classification and regression."
        https://www.sciencedirect.com/science/article/pii/S0893608003001163
    """
    def __init__(self, num: nn.Module, denom: nn.Module, bound_denom: bool=True, bound: float=0.1):
        super().__init__()
        assert bound > 0.0
        self.num = num
        self.denom = denom
        self.bound_denom = bound_denom
        self.bound = bound

    def forward(self, input: Tensor) -> Tensor:
        denom = self.denom(input)
        if self.bound_denom: denom = clamp(denom, min=self.bound)
        return self.num(input) / denom


class GeneralizedShuntingMLP(GeneralizedShuntingModule):
    def __init__(
            self, in_channels: int, out_channels: int,
            num_activation: nn.Module=nn.Identity(), denom_activation: nn.Module=nn.ReLU(),
            bound_denom: bool=True, bound: float=0.1
    ):
        num = nn.Sequential(nn.Linear(in_channels, out_channels, True), num_activation, BiasLayer((out_channels,)))
        denom = nn.Sequential(
            nn.Linear(in_channels, out_channels, True), denom_activation, BiasLayer((out_channels,), init=1.0)
        )
        super().__init__(num, denom, bound_denom, bound)
