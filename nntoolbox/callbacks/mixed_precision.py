"""
Implement mixed precision training as a callback
Adapted from fastai's course 2 v3 notebook and apex library
https://arxiv.org/pdf/1710.03740.pdf
"""

try:
    from apex.fp16_utils import convert_network, model_grads_to_master_grads, master_params_to_model_params
    from apex import amp
except:
    __all__ = []
else:
    import torch
    from torch.optim import Optimizer
    from nntoolbox.callbacks import Callback
    from typing import Dict, Any
    from torch import Tensor, float32, float16
    from typing import List, Tuple


    __all__ = ['MixedPrecision', 'MixedPrecisionV2']


    class MixedPrecision(Callback):
        """
        Callback for mixed precision training, adapted from fastai course 2 v3 notebook 10
        with minor changes to work with pytorch optimizer
        Training flow:
            switch model to float16, keeping a copy of params in float32
            forward (on float16 model):
                convert input to 16, forward all the way to prediction, convert back to 32, compute loss and scale
            backward:
                backprop through 16 model (float16 grad) -> dynamically adjust scaling factor
                (if no overflow) -> copy to float32 model -> scale down gradient -> update master model -> copy to float16 model
            switch model back to float32 after training
        """
        def __init__(
                self, loss_scale: int=512, dynamic: bool=True,
                max_loss_scale: float=2.**24, div_factor: float=2., scale_wait: int=500
        ):
            """
            :param loss_scale: loss scale (if not dynamic)
            :param dynamic: whether to scale loss dynamically
            :param max_loss_scale: upper bound for loss scale
            :param div_factor: how much to increment loss scale each time
            :param scale_wait: how many iterations of not overflowing to wait before scaling loss again
            """
            assert torch.backends.cudnn.enabled
            self.loss_scale = loss_scale if not dynamic else max_loss_scale
            self.max_loss_scale, self.div_factor, self.scale_wait = max_loss_scale, div_factor, scale_wait
            self.dynamic = dynamic
            self.learner = None

        def on_train_begin(self):
            """Convert network to float16"""
            self.learner._model = convert_network(self.learner._model, float16)
            self.model_param_groups, self.master_param_groups = get_param_groups(self.learner._optimizer)
            # self.learner._optimizer.param_groups = self.master_param_groups
            # self.learner._optimizer.zero_grad = self.learner._model.zero_grad
            copy_param_to_optimizer(self.learner._optimizer, self.master_param_groups)
            if self.dynamic: self.count = 0

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
            Scale the loss to prevent gradient underflow
            :param losses: dictionary of losses
            :return: scaled losses
            """
            if train:
                for key in losses:
                    losses[key] = losses[key] * self.loss_scale
            return losses

        def after_backward(self):
            """
            Copy the gradient to master and unscale
            :return: True if gradient does not overflow
            """

            if self.dynamic and check_grad_overflow(self.model_param_groups):
                # if overflow, divide the loss scale, zero grad and ignore batch:
                if self.loss_scale > 1:
                    self.loss_scale /= self.div_factor
                print("Overflow, changing loss scale to " + str(self.loss_scale))
                self.learner._model.zero_grad()
                self.count = 0
                return False

            to_master_grads(self.model_param_groups, self.master_param_groups)
            for group in self.master_param_groups:
                for param in group:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scale)

            if self.dynamic:
                self.count += 1
                if self.count >= self.scale_wait and self.loss_scale < self.max_loss_scale:
                    self.count = 0
                    self.loss_scale *= self.div_factor
                    print("Increasing loss scale to " + str(self.loss_scale))
            return True

        def after_step(self) -> bool:
            """
            Zero the gradient of the float16 and update master model's weight to float16 model
            :return: false (skipping zero grad step for optimizer)
            """
            self.learner._model.zero_grad()
            to_model_params(self.model_param_groups, self.master_param_groups)
            return False

        def on_batch_end(self, logs: Dict[str, Any]):
            if "loss" in logs:
                logs['loss'] /= self.loss_scale

        def on_train_end(self):
            """
            Convert model back to float
            """
            self.learner._model = self.learner._model.float()


    class MixedPrecisionV2(Callback):
        """
        Callback for mixed precision training to accomodate new apex api
        """
        def __init__(self, loss_scale: int=512, dynamic: bool=True):
            """
            :param loss_scale: loss scale (if not dynamic)
            :param dynamic: whether to scale loss dynamically
            """
            assert torch.backends.cudnn.enabled
            self.loss_scale = loss_scale
            self.dynamic = dynamic
            self.learner = None

        def on_train_begin(self):
            """Convert network to float16"""
            self.learner._model, self.optimizer = amp.initialize(
                self.learner._model, self.learner._optimizer,
                opt_level="O1", loss_scale='dynamic' if self.dynamic else self.loss_scale
            )
            self.learner._optimizer = DummyOptimizer()

        def after_losses(self, losses: Dict[str, Tensor], train: bool) -> Dict[str, Tensor]:
            """
            Call amp functionality
            """
            if train:
                total_loss = 0
                for key in losses:
                    total_loss += losses[key]
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
            return losses

        def on_backward_begin(self): return False

        def after_backward(self):
            self.optimizer.zero_grad()
            return True

        def after_step(self) -> bool: return False


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
        """
        Copy gradient from float 16 model to master model
        :param model_param_groups:
        :param master_param_groups:
        """
        for model_group, master_group in zip(model_param_groups, master_param_groups):
            model_grads_to_master_grads(model_params=model_group, master_params=master_group)


    def copy_param_to_optimizer(optimizer, param_groups):
        """
        Copy parameter to optimizer
        :param optimizer:
        :param param_groups:
        """
        for optimizer_group, model_group in zip(optimizer.param_groups, param_groups):
            optimizer_group['params'] = model_group


    def to_model_params(model_param_groups: List[List[Tensor]], master_param_groups: List[List[Tensor]]):
        """
        Copy master params to float16 model params
        :param model_param_groups:
        :param master_param_groups:
        :return:
        """
        for model_group, master_group in zip(model_param_groups, master_param_groups):
            master_params_to_model_params(model_params=model_group, master_params=master_group)


    def check_grad_overflow(param_groups: List[List[Tensor]]) -> bool:
        """
        Check whether any parameter's gradient is overflow
        :param param_groups:
        :return: true if a gradient is overflown
        """
        for group in param_groups:
            for param in group:
                if param.grad is not None:
                    grad_sum = float(param.grad.data.float().sum())
                    if grad_sum == float('-inf') or grad_sum == float('inf') or grad_sum != grad_sum: return True
        return False


    class DummyOptimizer:
        """
        Destroy original optimizer
        """
        def step(self): pass

        def zero_grad(self): pass
