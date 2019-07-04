"""
Implement mixed precision training as a callback
Based on fastai's notebook and apex library
"""
import torch
from nntoolbox.callbacks import Callback
# import apex.fp16_utils as fp16
from typing import Dict
from torch import Tensor, float32


class MixedPrecision(Callback):
    def __init__(self, loss_scale: int=512):
        self.loss_scale = loss_scale

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        for key in data:
            if data[key].dtype == float32:
                data[key] = data[key].half()
        return data


data = {"input_1": torch.rand(12).float(), "input_2": torch.rand(12).long()}
cb = MixedPrecision()
print(cb.on_batch_begin(data, True))

