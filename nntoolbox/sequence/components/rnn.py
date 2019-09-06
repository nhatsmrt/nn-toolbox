import torch
from torch import nn, Tensor
from typing import Tuple, Optional, Callable, Union, List
import torch.jit as jit
from collections import OrderedDict
from ...utils import dropout_mask


__all__ = [
    'RNNDropout', 'JitRNNDropout',
    'JitRNNLayer', 'JitLSTMLayer',
    'RNNSequential', 'JitRNNSequential',
    'JitResidualRNNV2', 'JitLSTMSequential', 'JitResidualLSTMV2'
]


class RNNDropout(nn.Module):
    def __init__(self, p):
        super(RNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, input_unpacked: Tensor, h_0: Tensor):
        return self.dropout(input_unpacked), h_0


class JitRNNDropout(jit.ScriptModule):
    def __init__(self, p):
        super(JitRNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    @jit.script_method
    def forward(self, input_unpacked: Tensor, h_0: Tensor) -> Tuple[Tensor, Tensor]:
        return self.dropout(input_unpacked), h_0


class JitRNNLayer(jit.ScriptModule):
    """
    Implement an RNN layer with jit script module

    References:

        The PyTorch Team. "Optimizing CUDA Recurrent Neural Networks with TorchScript."
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Yarin Gal, Zoubin Ghahramani. "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks."
        https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks
    """

    __constants__ = ['recurrent_drop_p', 'inp_drop_p']

    def __init__(
            self, base_cell: Callable[..., Union[jit.ScriptModule, nn.Module]], *cell_args,
            inp_drop_p: float=0.0, recurrent_drop_p: float=0.0
    ):
        assert 0.0 <= recurrent_drop_p < 1.0
        super().__init__()
        self.base_cell = base_cell(*cell_args)
        self.recurrent_drop_p = recurrent_drop_p
        self.inp_drop_p = inp_drop_p

    @jit.script_method
    def forward(self, input: Tensor, state: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        mask_r = jit.annotate(Tensor, torch.zeros(1))
        mask_i = jit.annotate(Tensor, torch.zeros(1))

        for t in range(len(inputs)):
            if self.training and self.inp_drop_p > 0.0:
                if t == 0:
                    mask_i = torch.rand(inputs[t].shape). \
                        bernoulli_(1 - self.recurrent_drop_p). \
                        div(1 - self.recurrent_drop_p)
                mask_i = mask_i.to(inputs[t].dtype).to(inputs[t].device)
                inputs[t] = mask_i * inputs[t]

            state = self.base_cell(inputs[t], state)
            outputs.append(state)

            if self.training and self.recurrent_drop_p > 0.0 and t < len(inputs) - 1:
                if t == 0:
                    mask_r = torch.rand(state.shape). \
                        bernoulli_(1 - self.recurrent_drop_p). \
                        div(1 - self.recurrent_drop_p)
                mask_r = mask_r.to(state.dtype).to(state.device)
                state = mask_r * state

        return torch.stack(outputs, dim=0), state


class JitLSTMLayer(jit.ScriptModule):
    """
    Implement an LSTM layer with jit script module

    References:

        The PyTorch Team. "Optimizing CUDA Recurrent Neural Networks with TorchScript."
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Yarin Gal, Zoubin Ghahramani. "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks."
        https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks
    """

    __constants__ = ['recurrent_drop_p', 'inp_drop_p']

    def __init__(
            self, base_cell: Callable[..., Union[jit.ScriptModule, nn.Module]], *cell_args,
            inp_drop_p: float=0.0, recurrent_drop_p: float=0.0
    ):
        super().__init__()
        self.recurrent_drop_p = recurrent_drop_p
        self.base_cell = base_cell(*cell_args)
        self.inp_drop_p = inp_drop_p

    @jit.script_method
    def forward(
            self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        mask_r = jit.annotate(Tensor, torch.zeros(1))
        mask_i = jit.annotate(Tensor, torch.zeros(1))

        for t in range(len(inputs)):
            if self.training and self.inp_drop_p > 0.0:
                if t == 0:
                    mask_i = torch.rand(inputs[t].shape). \
                        bernoulli_(1 - self.recurrent_drop_p). \
                        div(1 - self.recurrent_drop_p)
                mask_i = mask_i.to(inputs[t].dtype).to(inputs[t].device)
                inputs[t] = mask_i * inputs[t]

            output, state = self.base_cell(inputs[t], state)
            outputs.append(output)

            if self.training and self.recurrent_drop_p > 0.0 and t < len(inputs) - t:
                if t == 0:
                    mask_r = torch.rand(state[0].shape) \
                        .bernoulli_(1 - self.recurrent_drop_p) \
                        .div(1 - self.recurrent_drop_p)
                mask_r = mask_r.to(state[0].dtype).to(state[0].device)
                state[0] = mask_r * state[0]

        return torch.stack(outputs, dim=0), state


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


class JitResidualRNNV2(jit.ScriptModule):
    """
    Implement a simple residual stacked RNN

    References:

        Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi.
        "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation."
        https://arxiv.org/pdf/1609.08144.pdf
    """
    __constants__ = ['_modules_list', 'skip_length']

    def __init__(self, layers: List[Union[jit.ScriptModule, nn.Module]], skip_length: int=1):
        """
        :param layers: rnn layers. Must have output dimension the same as input dimension
        :param skip_length: length of the skip
        """
        super(JitResidualRNNV2, self).__init__()
        assert skip_length > 0
        self._modules_list = nn.ModuleList(layers)
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


class JitResidualLSTMV2(jit.ScriptModule):
    """
    Implement a simple residual stacked LSTM

    References:

        Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi.
        "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation."
        https://arxiv.org/pdf/1609.08144.pdf
    """
    __constants__ = ['_modules_list', 'skip_length']

    def __init__(self, layers: List[Union[jit.ScriptModule, nn.Module]], skip_length: int=1):
        """
        :param layers: rnn layers. Must have output dimension the same as input dimension
        :param skip_length: length of the skip
        """
        super().__init__()
        self._modules_list = nn.ModuleList(layers)
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
