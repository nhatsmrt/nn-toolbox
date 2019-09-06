from torch import nn, Tensor
from torch import jit
import torch
from typing import Tuple, Callable, Union, List, Optional
from nntoolbox.init import sqrt_uniform_init
from nntoolbox.sequence.components import \
    MIGRUCell, RecurrentHighwayCell, MILSTMCell, MultiplicativeRNNCell, JitLSTMLayer, \
    JitRNNLayer, JitRNNSequential, JitResidualRNNV2, ONLSTMCell, ONLSTMCellV2, \
    JitLSTMSequential, JitResidualLSTMV2
from nntoolbox.hooks import OutputHook
from nntoolbox.components import DropConnect
# from nntoolbox.components import LWTA

# print(MILSTMCell(input_size=128, hidden_size=256))

# cell = MILSTMCell(input_size=128, hidden_size=256)
# inp = torch.rand(32, 128)
# h0, c0 = torch.zeros(32, 256), torch.zeros(32, 256)
# op, (h1, c1) = cell(inp, (h0, c0))
# print(op.shape)
# print(h1.shape)
# print(c1.shape)


# class FastWeightRNNCell(jit.ScriptModule):
#     def __init__(self, input_size: int, hidden_size: int):
#         super().__init__()


# cell = MultiplicativeRNNCell(input_size=128, hidden_size=256, intermediate_size=64)
# DropConnectFast = lambda *args, **kwargs: jit.script(JitDropConnect(*args, **kwargs))
# cell = JitDropConnect(RecurrentHighwayCell(128, 256, 1), [0.5], ["weight_s"])
# # cell = ONLSTMCell(128, 256)
# inp = torch.rand(32, 128)
# h0 = torch.zeros(32, 256)
# # print(cell(inp, h0).shape)
# target = torch.rand(32, 256)
# optimizer = torch.optim.Adam(cell.parameters())
# loss = nn.MSELoss()
#
#
# for _ in range(1000):
#     optimizer.zero_grad()
#     l = loss(cell(inp, h0), target)
#     l.backward()
#     optimizer.step()
#     print(l)


# layer = JitRNNLayer(RecurrentHighwayCell, 128, 256, 3, recurrent_drop_p=0.1, inp_drop_p=0.1)
layer = JitLSTMLayer(MILSTMCell, 128, 256, recurrent_drop_p=0.1, inp_drop_p=0.1)
# layer = JitRNNLayer(MIGRUCell, 128, 256)
# layer = JitRNNSequential(
#     layers=[JitRNNLayer(MIGRUCell, 128, 256)] +
#            [JitResidualRNNV2([JitRNNLayer(MIGRUCell, 256, 256) for _ in range(2)])]
# )
# from functools import partial
# layer = JitResidualRNN(
#     base_rnn=partial(JitRNNLayer, MIGRUCell),
#     bidirectional=False,
#     input_size=128,
#     num_layers=3
# )

# layer = nn.Sequential(layer)
# hook = OutputHook(layer)
inp = torch.rand(7, 32, 128)
# state = torch.zeros((inp.shape[1], 256)).to(inp.device).to(inp.dtype)
target = torch.rand(7, 32, 256)
optimizer = torch.optim.Adam(layer.parameters())
loss = nn.MSELoss()
#
for _ in range(10000):
    optimizer.zero_grad()
    l = loss(layer(inp)[0], target)
    l.backward()
    # print(next(layer.parameters()).grad)
    optimizer.step()
    print(l)

print(layer(inp)[0].shape)


