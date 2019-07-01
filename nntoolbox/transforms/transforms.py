from torch import Tensor
from numpy import ndarray


__all__ = ['ToNumpyArray']


class ToNumpyArray:
    def __call__(self, input: Tensor) -> ndarray:
        return input.cpu().detach().numpy()
