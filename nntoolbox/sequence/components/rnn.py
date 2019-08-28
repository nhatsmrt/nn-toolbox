import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from typing import Tuple, Optional, Callable, Union, List
import torch.jit as jit
from collections import OrderedDict


__all__ = [
    'RNNDropout', 'RNNSequential', 'JitRNNDropout', 'JitRNNSequential',
    'JitRNNLayer', 'JitLSTMLayer'
]


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


class JitRNNDropout(jit.ScriptModule):
    def __init__(self, p):
        super(JitRNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    @jit.script_method
    def forward(self, input_unpacked: Tensor, h_0: Tensor) -> Tuple[Tensor, Tensor]:
        return self.dropout(input_unpacked), h_0


class JitRNNSequential(jit.ScriptModule):
    """
    Implement a simple stacked RNN
    """
    __constants__ = ['_modules_list']

    def __init__(self, layers: List[Union[jit.ScriptModule, nn.Module]]):
        super(JitRNNSequential, self).__init__()
        self._modules_list = nn.ModuleList(layers)

    @jit.script_method
    def forward(self, input: Tensor, h_0: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        for module in self._modules_list:
            input, h_0 = module(input, h_0)
        return input, h_0


class JitRNNLayer(jit.ScriptModule):
    """
    Implement an RNN layer with jit script module

    References:

        The PyTorch Team. "Optimizing CUDA Recurrent Neural Networks with TorchScript."
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """
    def __init__(self, base_cell: Callable[..., Union[jit.ScriptModule, nn.Module]], *cell_args):
        super().__init__()
        self.base_cell = base_cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])

        for t in range(len(inputs)):
            state = self.base_cell(inputs[t], state)
            outputs.append(state)

        return torch.stack(outputs, dim=0), state


class JitLSTMLayer(JitRNNLayer):
    """
    Implement an LSTM layer with jit script module

    References:

        The PyTorch Team. "Optimizing CUDA Recurrent Neural Networks with TorchScript."
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """
    @jit.script_method
    def forward(
            self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])

        for t in range(len(inputs)):
            output, state = self.base_cell(inputs[t], state)
            outputs.append(output)

        return torch.stack(outputs, dim=0), state


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


