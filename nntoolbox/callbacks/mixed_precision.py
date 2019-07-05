"""
Implement mixed precision training as a callback
Based on fastai's course 2 v3 notebook and apex library
"""
import torch
from torch.optim import Optimizer
from nntoolbox.callbacks import Callback
from apex.fp16_utils import convert_network, model_grads_to_master_grads, master_params_to_model_params
from typing import Dict
from torch import Tensor, float32, float16
from typing import List, Tuple


__all__ = ['MixedPrecision']


class MixedPrecision(Callback):
    """
    Callback for mixed precision training, based on fastai course 2 v3 notebook 10
    Training flow:
        switch model to float16, keeping a copy of params in float32
        forward (on float16 model):
            convert input to 16, forward all the way to prediction, convert back to 32, compute loss and scale
        backward:
            backprop through 16 model (float16 grad) -> copy to float32 model -> scale down gradient
            -> update master model -> copy to float16 model
        switch model back to float32 after training
    """
    def __init__(self, loss_scale: int=512):
        assert torch.backends.cudnn.enabled
        self.loss_scale = loss_scale
        self.learner = None

    def on_train_begin(self):
        """Convert network to float16"""
        self.learner._model = convert_network(self.learner._model, float16)
        self.model_param_groups, self.master_param_groups = get_param_groups(self.learner._optimizer)
        # self.learner._optimizer.param_groups = self.master_param_groups
        # self.learner._optimizer.zero_grad = self.learner._model.zero_grad
        copy_param_to_optimizer(self.learner._optimizer, self.master_param_groups)

    def on_batch_begin(self, data: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
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
        """
        Convert the output to float32 before computing loss
        :param outputs: dictionary of outputs
        :param train: whether is training
        :return: converted outputs
        """
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

    def after_backward(self):
        """
        Copy the gradient to master and unscale
        """
        to_master_grads(self.model_param_groups, self.master_param_groups)
        for group in self.master_param_groups:
            for param in group:
                if param.grad is not None: param.grad.div_(self.loss_scale)

    def after_step(self) -> bool:
        """
        Zero the gradient of the float16 and update master model's weight to float16 model
        :return:
        """
        self.learner._model.zero_grad()
        to_model_params(self.model_param_groups, self.master_param_groups)
        return False

    def on_train_end(self):
        """
        Convert model back to float
        """
        self.learner._model = self.learner._model.float()


def get_param_groups(optimizer: Optimizer) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    """
    Store lists (grouped) params of a float16 model and its float32 version
    :param optimizer:
    :return: lists of params
    """
    model_param_groups = [[param for param in group['params'] if param.requires_grad] for group in optimizer.param_groups]
    master_param_groups = [
        [param.clone().float().detach() for param in group['params'] if param.requires_grad]
        for group in optimizer.param_groups
    ]
    for group in master_param_groups:
        for param in group:
            param.requires_grad_(True)
    return model_param_groups, master_param_groups


def to_master_grads(model_param_groups: List[List[Tensor]], master_param_groups: List[List[Tensor]]):
    for model_group, master_group in zip(model_param_groups, master_param_groups):
        model_grads_to_master_grads(model_params=model_group, master_params=master_group)


def copy_param_to_optimizer(optimizer, param_groups):
    for optimizer_group, model_group in zip(optimizer.param_groups, param_groups):
        optimizer_group['params'] = model_group


def to_model_params(model_param_groups: List[List[Tensor]], master_param_groups: List[List[Tensor]]):
    for model_group, master_group in zip(model_param_groups, master_param_groups):
        master_params_to_model_params(model_params=model_group, master_params=master_group)




# layer = torch.nn.Linear(3, 5)
# optimizer = torch.optim.Adam(layer.parameters())
# for group in optimizer.param_groups:
#     for param in group['params']:
#         print(param.requires_grad)