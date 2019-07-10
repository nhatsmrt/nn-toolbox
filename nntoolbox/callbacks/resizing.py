from .callbacks import Callback
from typing import Dict, Any
from torch import Tensor
import torch.nn.functional as F


__all__ = ['InputProgressiveResizing']


# UNTESTED
class InputProgressiveResizing(Callback):
    """
    Implement a callback for progressive resizing (input only)
    """
    def __init__(self, initial_size: int, max_size: int, upscale_every: int, upscale_factor: float, mode='bilinear'):
        self.size, self.max_size = initial_size, max_size
        self.initial_size = initial_size
        self.upscale_every, self.upscale_factor = upscale_every, upscale_factor
        self.mode = mode

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        data["inputs"] = F.interpolate(data["inputs"], size=(self.size, self.size), mode=self.mode)
        return data

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if logs["epoch"] % self.upscale_every == 0 and self.size * self.upscale_factor <= self.max_size:
            self.size = int(self.initial_size * (self.upscale_factor ** (logs["epoch"] // self.upscale_every)))
            print("Increasing the scale of input to " + str(self.size))
        return False
