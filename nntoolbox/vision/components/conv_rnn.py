import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Union

__all__ = ['ConvLSTM2dCell', 'ConvLSTM2d']


class ConvLSTM2dCell(nn.Module):
    """
    Analogous to LSTM cell, but replaces the linear transformation in the gates' definition with a convolutional layer.
    For simplicity and efficiency reason, assumes that hidden state's spatial dimension is the same as that of input;
    'same' padding will be enforced.

    References:
    Xingjian Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting."
    https://arxiv.org/abs/1506.04214
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: Union[int, Tuple[int, int]], **kwargs):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(in_channels + hidden_channels, hidden_channels * 4, kernel_size, padding='same', **kwargs)

    def forward(self, inputs: Tensor, hc: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tensor]:
        if hc is None:
            hidden_dimensions = inputs.shape[0], self.hidden_channels, inputs.shape[2], inputs.shape[3]

            hidden = torch.zeros(*hidden_dimensions)
            cell = torch.zeros(*hidden_dimensions)
        else:
            hidden, cell = hc

        gate_inp = torch.cat([inputs, hidden], dim=1)
        inp_gate, forget_gate, inter_cell, out_gate = self.gates(gate_inp).chunk(4, 1)

        cell_ret = torch.sigmoid(forget_gate) * cell + torch.sigmoid(inp_gate) * torch.tanh(inter_cell)
        hidden_ret = torch.sigmoid(out_gate) * torch.tanh(cell_ret)

        return (hidden_ret, cell_ret)


class ConvLSTM2d(nn.Module):
    # TODO: support batch_first == True, num_layer and bidirectional
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: Union[int, Tuple[int, int]], **kwargs):
        """
        Analogous to LSTM, but replaces the linear transformation in the gates' definition with a convolutional layer.
        For simplicity and efficiency reason, assumes that hidden state's spatial dimension is the same as that of input;
        'same' padding will be enforced.
        Only supports 1 layer, single direction LSTM for now.

        References:
        Xingjian Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting."
        https://arxiv.org/abs/1506.04214
        """
        super().__init__()
        self.cell = ConvLSTM2dCell(in_channels, hidden_channels, kernel_size, **kwargs)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        outputs = []
        hc = None

        for t in range(inputs.shape[0]):
            input = inputs[0]
            hc = self.cell(input, hc)
            outputs.append(hc[0])

        output = torch.stack(outputs, dim=0)
        return output, hc
