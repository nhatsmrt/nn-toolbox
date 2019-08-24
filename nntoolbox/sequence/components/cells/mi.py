import torch
from torch import nn, Tensor
from typing import Tuple
from torch import jit


__all__ = ['MILSTMCell', 'MIGRUCell']


class MILSTMCell(jit.ScriptModule):
    """
    Multiplicative Integration LSTM Cell

    References:

        Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, Ruslan Salakhutdinov.
        "On Multiplicative Integration with Recurrent Neural Networks."
        https://arxiv.org/abs/1606.06630

        The PyTorch Team. "Optimizing CUDA Recurrent Neural Networks with TorchScript."
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_mult = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_ind = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        input_transform = torch.mm(input, self.weight_ih.t())
        hidden_transform = torch.mm(hx, self.weight_hh.t())
        gates = (
            input_transform * hidden_transform * self.bias_mult
            + input_transform * self.bias_ih
            + hidden_transform * self.bias_hh
            + self.bias_ind
        )

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate, forgetgate, cellgate, outgate = \
            torch.sigmoid(ingate), torch.sigmoid(forgetgate), torch.tanh(cellgate), torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class MIGRUCell(jit.ScriptModule):
    """
    Multiplicative Integration GRU Cell

    References:

        Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, Ruslan Salakhutdinov.
        "On Multiplicative Integration with Recurrent Neural Networks."
        https://arxiv.org/abs/1606.06630

        The PyTorch Team. "Optimizing CUDA Recurrent Neural Networks with TorchScript."
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ig = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hg = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.bias_ig = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_hg = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_mult_g = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_ind_g = nn.Parameter(torch.randn(2 * hidden_size))

        self.weight_ic = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hc = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(hidden_size))
        self.bias_hc = nn.Parameter(torch.randn(hidden_size))
        self.bias_mult_c = nn.Parameter(torch.randn(hidden_size))
        self.bias_ind_c = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        input_transform = torch.mm(input, self.weight_ig.t())
        hidden_transform = torch.mm(hx, self.weight_hg.t())
        gates = (
            input_transform * hidden_transform * self.bias_mult_g
            + input_transform * self.bias_ig
            + hidden_transform * self.bias_hg
            + self.bias_ind_g
        )
        update_gate, reset_gate = gates.chunk(2, 1)

        input_transform_c = torch.mm(input, self.weight_ic.t())
        hidden_transform_c = torch.mm(reset_gate * hx, self.weight_hc.t())
        candidate = torch.tanh(
            input_transform_c * hidden_transform_c * self.bias_mult_c
            + input_transform_c * self.bias_ic
            + hidden_transform_c * self.bias_hc
            + self.bias_ind_c
        )

        return (1.0 - update_gate) * hx + update_gate * candidate
