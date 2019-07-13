import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from typing import Tuple
import torch.jit as jit
from collections import OrderedDict


__all__ = ['RNNDropout', 'RNNSequential', 'FastRNNDropout', 'FastRNNSequential']


class RNNDropout(nn.Module):
    def __init__(self, p):
        super(RNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, input_unpacked: Tensor, h_0: Tensor):
        return self.dropout(input_unpacked), h_0


class RNNSequential(nn.Module):
    def __init__(self, *layers):
        super(RNNSequential, self).__init__()
        if len(layers) == 1 and isinstance(layers[0], OrderedDict):
            for key, module in layers[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(layers):
                self.add_module(str(idx), module)

    def forward(self, input, h_0):
        for module in self._modules.values():
            input, h_0 = module(input, h_0)
        return input, h_0


class FastRNNDropout(jit.ScriptModule):
    def __init__(self, p):
        super(FastRNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    @jit.script_method
    def forward(self, input_unpacked: Tensor, h_0: Tensor) -> Tuple[Tensor, Tensor]:
        return self.dropout(input_unpacked), h_0


class FastRNNSequential(jit.ScriptModule):
    __constants__ = ['_modules_list']

    def __init__(self, *layers):
        super(FastRNNSequential, self).__init__()
        self._modules_list = nn.ModuleList(layers)

    #         if len(layers) == 1 and isinstance(layers[0], OrderedDict):
    #             for key, module in layers[0].items():
    #                 self.add_module(key, module)
    #         else:
    #             for idx, module in enumerate(layers):
    #                 self.add_module(str(idx), module)

    @jit.script_method
    def forward(self, input: Tensor, h_0: Tensor) -> Tuple[Tensor, Tensor]:
        for module in self._modules_list:
            input, h_0 = module(input, h_0)
        return input, h_0


# INCOMPLETE
class ReverseRNN(nn.Module):
    """
    Explicitly encode RNN for reverse direction (INCOMPLETE)
    """
    def __init__(self, base_rnn, num_layers: int, input_size: int, hidden_size: int, dropout: float = 0.0, **kwargs):
        super(ReverseRNN, self).__init__()
        self.base_rnn = base_rnn(
            num_layers=1, input_size=input_size, hidden_size=hidden_size,
            bidirectional=False, **kwargs
        )

    def forward(self, input, h_0, lengths: None):
        if isinstance(input, PackedSequence):
            input_unpacked, lengths = pad_packed_sequence(input)
            input_packed = input
        elif isinstance(input, Tensor):
            assert lengths is not None
            assert len(input.shape) == 3

            input_unpacked = input
            input_packed = pack_padded_sequence(input, lengths=lengths)
        else:
            raise ValueError

        if h_0 is None:
            h_0 = torch.zeros(
                size=(self._num_layers, input_unpacked.shape[1], input_unpacked.shape[2])
            ).to(input_unpacked.device)

        output, h_n = self.base_rnn(input_packed, h_0)
        return


