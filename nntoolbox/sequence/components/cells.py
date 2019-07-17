import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Union, Tuple, Iterable


__all__ = ['ZoneoutCell']


class ZoneoutCell(nn.Module):
    """
    Implement zoneout: randomly preserve old hidden state values

    Reference: https://arxiv.org/pdf/1606.01305.pdf

    WARNING: SLOW (NOT IMPLEMENTED WITH JIT); UNTESTED
    """
    def __init__(self, base_cell: nn.Module, p: Union[float, Iterable[float]]):
        assert 0 <= p <= 1
        super(ZoneoutCell, self).__init__()
        self.base_cell = base_cell
        if isinstance(p, float):
            p = [p]
        self.p = list(p)

    def forward(self, input: Tensor, state: Optional[Union[Tensor, Tuple[Tensor]]]=None) -> Union[Tensor, Tuple[Tensor]]:
        output = self.base_cell(input, state)
        if self.training:
            if isinstance(output, tuple):
                if len(self.p) == 1:
                    self.p = [self.p[0] for _ in range(len(output))]
                new_outputs = []
                for i in range(len(output)):
                    mask = self.get_mask(output[i], self.p[i])
                    if state is None:
                        new_outputs.append(output[i] * mask)
                    else:
                        new_outputs.append(output[i] * mask + (1 - mask) * state[i])
                return tuple(output)
            else:
                mask = self.get_mask(output, self.p[0])
                if state is None:
                    return mask * output
                else:
                    return output * mask + (1 - mask) * state
        else:
            return output

    def get_mask(self, tensor: Tensor, p: float) -> Tensor:
        return torch.zeros(tensor.shape).bernoulli_(p).to(tensor.dtype).to(tensor.device)
