from torch import nn, Tensor
from ....init import sqrt_uniform_init
from torch import jit
import torch
from typing import Optional


__all__ = ['RecurrentHighwayCell']


class RecurrentHighwayCell(jit.ScriptModule):
    """
    Implement a recurrence highway cell for deep recurrence:

    h^l(t) = tanh( W_h x(t) I_{l = 0} + R_hl s^{l-1}(t) + b_hl) (transformed)

    t^l(t) = tanh( W_t x(t) I_{l = 0} + R_tl s^{l-1}(t) + b_tl) (transform gate)

    c^l(t) = tanh( W_c x(t) I_{l = 0} + R_cl s^{l-1}(t) + b_cl) (carry gate)

    s^l(t) = h^l(t) * t^l(t) + c^l(t) s^{l-1}(t) (state)

    s^{-1}(t) = y(t - 1) = s^L(t - 1)

    References:

        Julian Georg Zilly, Rupesh Kumar Srivastava, Jan Koutník, Jürgen Schmidhuber. "Recurrent Highway Networks."
        http://proceedings.mlr.press/v70/zilly17a/zilly17a.pdf
    """

    __constants__ = ['recurrence_depth', 'state_size']

    def __init__(self, input_size: int, state_size: int, recurrence_depth: int):
        super().__init__()
        self.state_size = state_size
        self.weight_i = nn.Parameter(torch.rand(3 * state_size, input_size))
        self.weight_s = nn.Parameter(torch.rand(recurrence_depth, 3 * state_size, state_size))
        self.bias = nn.Parameter(torch.rand(recurrence_depth, 3 * state_size))
        self.recurrence_depth = recurrence_depth
        sqrt_uniform_init(self)

    @jit.script_method
    def forward(self, input: Tensor, state: Optional[Tensor]=None) -> Tensor:
        if state is None: state = torch.zeros((input.shape[0], self.state_size)).to(input.device).to(input.dtype)

        for l in range(self.recurrence_depth):
            if l == 0:
                input_state = torch.cat([input, state], dim=-1)
                weight = torch.cat([self.weight_i, self.weight_s[l]], dim=-1)
                gates = torch.mm(input_state, weight.t()) + self.bias[l]
            else:
                gates = torch.mm(state, self.weight_s[l].t()) + self.bias[l]

            h, t, c = gates.chunk(3, 1)
            h, t, c = torch.tanh(h), torch.sigmoid(t), torch.sigmoid(c)
            state = h * t + c * state

        return state
