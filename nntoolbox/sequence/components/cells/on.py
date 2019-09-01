from torch import nn, Tensor
from torch import jit
import torch
from typing import Tuple, Optional
from nntoolbox.init import sqrt_uniform_init


__all__ = ['Cumax', 'ONLSTMCell', 'ONLSTMCellV2']


class Cumax(jit.ScriptModule):
    def forward(self, input: Tensor) -> Tensor:
        return torch.cumsum(torch.softmax(input, dim=-1), dim=-1)


class ONLSTMCell(jit.ScriptModule):
    """
    Ordered Neuron LSTM. Augmenting LSTM with the hierarchical inductive bias by ordering the neurons of each hidden
    states. This is the recommended version.

    References:

        Yikang Shen, Shawn Tan, Alessandro Sordoni, Aaron Courville.
        "Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks."
        https://openreview.net/forum?id=B1l6qiR5F7
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.weight_i = nn.Parameter(torch.rand((hidden_dim * 6, input_dim)))
        self.weight_h = nn.Parameter(torch.rand((hidden_dim * 6, hidden_dim)))
        self.bias = nn.Parameter(torch.rand(hidden_dim * 6))
        sqrt_uniform_init(self)
        self.cumax = Cumax()

    @jit.script_method
    def forward(
            self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if state is None: state = (
            torch.zeros((input.shape[0], self.weight_h.shape[1])).to(input.device).to(input.dtype),
            torch.zeros((input.shape[0], self.weight_h.shape[1])).to(input.device).to(input.dtype)
        )

        hx, cx = state
        gates = (torch.mm(input, self.weight_i.t()) + torch.mm(input, self.weight_i.t()) + self.bias).chunk(6, 1)

        master_f, master_i, f, i, o, c = self.cumax(gates[0]), 1.0 - self.cumax(gates[1]), torch.sigmoid(gates[2]), \
                                         torch.sigmoid(gates[3]), torch.sigmoid(gates[4]), torch.tanh(gates[5])

        overlap = master_f * master_i
        f_hat = f * overlap + master_f - overlap
        i_hat = i * overlap + master_i - overlap

        cy = f_hat * cx + i_hat * c
        hy = o * torch.tanh(cy)
        return hy, (hy, cy)


class ONLSTMCellV2(jit.ScriptModule):
    """
    Ordered Neuron LSTM. Augmenting LSTM with the hierarchical inductive bias by ordering the neurons of each hidden
    states. This is NOT the recommended version.

    References:

        Yikang Shen, Shawn Tan, Alessandro Sordoni, Aaron Courville.
        "Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks."
        https://openreview.net/forum?id=B1l6qiR5F7
    """

    __constants__ = ['hidden_dim']

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim * 6)
        self.hidden_dim = hidden_dim
        self.cumax = Cumax()

    @jit.script_method
    def forward(
            self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if state is None: state = (
            torch.zeros((input.shape[0], self.hidden_dim)).to(input.device).to(input.dtype),
            torch.zeros((input.shape[0], self.hidden_dim)).to(input.device).to(input.dtype)
        )

        hx, cx = state
        gates = self.linear(torch.cat((input, hx), dim=-1)).chunk(6, 1)

        master_f, master_i, f, i, o, c = self.cumax(gates[0]), 1.0 - self.cumax(gates[1]), torch.sigmoid(gates[2]), \
                                         torch.sigmoid(gates[3]), torch.sigmoid(gates[4]), torch.tanh(gates[5])

        overlap = master_f * master_i
        f_hat = f * overlap + master_f - overlap
        i_hat = i * overlap + master_i - overlap

        cy = f_hat * cx + i_hat * c
        hy = o * torch.tanh(cy)
        return hy, (hy, cy)
