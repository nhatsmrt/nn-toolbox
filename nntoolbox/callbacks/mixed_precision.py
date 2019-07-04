"""
Implement mixed precision training as a callback
Based on fastai's notebook and apex library
"""
import torch
from nntoolbox.callbacks import Callback
import apex.fp16_utils as fp16
from typing import Dict
from torch import Tensor, float32, float16


class MixedPrecision(Callback):
    def __init__(self, learner, loss_scale: int=512):
        self.loss_scale = loss_scale
        self.learner = learner

    def on_train_begin(self):
        """Convert network to float16"""
        self.learner._model = fp16.convert_network(self.learner._model, float16)

    def on_batch_begin(self, data: Dict[str, Tensor], train) -> Dict[str, Tensor]:
        """
        Convert all float32 data to float16
        :param data: data of float
        :param train: whether is training
        :return: converted data
        """
        for key in data:
            if data[key].dtype == float32:
                data[key] = data[key].half()
        return data

    def after_outputs(self, outputs: Dict[str, Tensor], train: bool):
        for key in outputs:
            if outputs[key].dtype == float16:
                outputs[key] = outputs[key].float()
        return outputs

    def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
        """
        Scale the loss to prevent gradient vanishing
        :param losses: dictionary of losses
        :return: scaled losses
        """
        for key in losses:
            losses[key] = losses[key] * self.loss_scale
        return losses

    def on_train_end(self):
        """
        Convert model back to float
        """
        self.learner._model = self.learner._model.float()


data = {"input_1": torch.rand(12).float(), "input_2": torch.rand(12).long()}
# cb = MixedPrecision()
print(cb.on_batch_begin(data, True))
