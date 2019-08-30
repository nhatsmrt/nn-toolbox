import torch
from torch import nn, Tensor
from typing import Tuple, Optional
from ....init import sqrt_uniform_init
from torch import jit


__all__ = ['MultiplicativeRNNCell', 'MILSTMCell', 'MIGRUCell']


class MultiplicativeRNNCell(jit.ScriptModule):
    """
    Multiplicative RNN. Allowing input to change the hidden state easier by introducing multiplicative interaction:

    m_t = (W_mx x_t) * (W_mh h_{t-1})

    h_t = tanh(W_hm m_t + W_hx x_t)

    Note that the implementation is based on the re-formulation from the second reference.

    References:
        Ilya Sutskever, James Martens, and Geoffrey Hinton. "Generating Text with Recurrent Neural Networks."
        https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf

        Ben Krause, Iain Murray, Steve Renals, and Liang Lu. "MULTIPLICATIVE LSTM FOR SEQUENCE MODELLING."
        https://arxiv.org/pdf/1609.07959.pdf
    """

    __constants__ = ['intermediate_size', 'hidden_size', 'bias']

    def __init__(self, input_size: int, hidden_size: int, intermediate_size: int, bias: bool=True):
        super().__init__()
        self.hidden_size, self.intermediate_size, self.bias = hidden_size, intermediate_size, bias
        self.weight_i = nn.Parameter(torch.rand(intermediate_size + hidden_size, input_size))
        self.weight_h = nn.Parameter(torch.rand(intermediate_size, hidden_size))
        self.weight_m = nn.Parameter(torch.rand(hidden_size, intermediate_size))

        if bias:
            self.bias_i = nn.Parameter(torch.rand(intermediate_size + hidden_size))
            self.bias_h = nn.Parameter(torch.rand(intermediate_size))
            self.bias_op = nn.Parameter(torch.rand(hidden_size))

        sqrt_uniform_init(self)

    @jit.script_method
    def forward(self, input: Tensor, state: Optional[Tensor]=None) -> Tensor:
        if state is None: state = torch.zeros((input.shape[0], self.hidden_size)).to(input.device).to(input.dtype)
        input_trans = torch.matmul(input, self.weight_i.t())
        if self.bias: input_trans += self.bias_i

        intermediate = input_trans[:, :self.intermediate_size] * torch.matmul(state, self.weight_h.t())
        if self.bias: intermediate += self.bias_h

        op = torch.matmul(intermediate, self.weight_m.t()) + input_trans[:, self.intermediate_size:]
        if self.bias: op += self.bias_op
        return torch.tanh(op)


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

    __constants__ = ['hidden_size']

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
        sqrt_uniform_init(self)

    @jit.script_method
    def forward(
            self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if state is None: state = (
            torch.zeros((input.shape[0], self.hidden_size)).to(input.device).to(input.dtype),
            torch.zeros((input.shape[0], self.hidden_size)).to(input.device).to(input.dtype)
        )

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

    __constants__ = ['hidden_size']

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
        sqrt_uniform_init(self)

    @jit.script_method
    def forward(self, input: Tensor, hx: Optional[Tensor]=None) -> Tensor:
        if hx is None: hx = torch.zeros((input.shape[0], self.hidden_size)).to(input.device).to(input.dtype)

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
