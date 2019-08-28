import torch
from torch import nn, Tensor
from typing import Tuple, Optional, Callable, Union, List
import torch.jit as jit
from collections import OrderedDict


__all__ = [
    'JitRNNLayer', 'JitLSTMLayer', 'RNNDropout',
    'RNNSequential', 'JitRNNDropout', 'JitRNNSequential',
    'JitResidualRNNV2', 'JitLSTMSequential', 'JitResidualLSTMV2'
]


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


class JitResidualRNNV2(JitRNNSequential):
    """
    Implement a simple residual stacked RNN
    """
    __constants__ = ['_modules_list', 'skip_length']

    def __init__(self, layers: List[Union[jit.ScriptModule, nn.Module]], skip_length: int=1):
        """
        :param layers: rnn layers. Must have output dimension the same as input dimension
        :param skip_length: length of the skip
        """
        super(JitResidualRNNV2, self).__init__(layers)
        assert skip_length > 0
        self.skip_length = skip_length

    @jit.script_method
    def forward(self, input: Tensor, h_0: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        output = jit.annotate(Tensor, torch.zeros(1))
        l = 1
        for module in self._modules_list:
            output, h_0 = module(input, h_0)
            if l % self.skip_length == 0:
                input = output + input
            else:
                input = output
            l += 1
        return output, h_0


class JitLSTMSequential(jit.ScriptModule):
    """
    Implement a simple stacked LSTM
    """
    __constants__ = ['_modules_list']

    def __init__(self, layers: List[Union[jit.ScriptModule, nn.Module]]):
        super(JitLSTMSequential, self).__init__()
        self._modules_list = nn.ModuleList(layers)

    @jit.script_method
    def forward(
            self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        for module in self._modules_list:
            input, states = module(input, states)
        return input, states


class JitResidualLSTMV2(JitLSTMSequential):
    """
    Implement a simple residual stacked RNN
    """
    __constants__ = ['_modules_list', 'skip_length']

    def __init__(self, layers: List[Union[jit.ScriptModule, nn.Module]], skip_length: int=1):
        """
        :param layers: rnn layers. Must have output dimension the same as input dimension
        :param skip_length: length of the skip
        """
        super().__init__(layers)
        assert skip_length > 0
        self.skip_length = skip_length

    @jit.script_method
    def forward(
            self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        output = jit.annotate(Tensor, torch.zeros(1))
        l = 1
        for module in self._modules_list:
            output, states = module(input, states)
            if l % self.skip_length == 0:
                input = output + input
            else:
                input = output
            l += 1
        return output, states
