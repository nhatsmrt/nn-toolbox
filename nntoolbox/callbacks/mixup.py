from typing import Sequence, Tuple, Dict, Any
from .callbacks import Callback
# from ..vision.learner import SupervisedImageLearner
import numpy as np
from torch import Tensor, nn
import torch


__all__ = ['ManifoldMixupCallback', 'ManifoldMixupModule']


class ManifoldMixupModule(nn.Module):
    """
    Wrapper module to apply manifold mixup
    """
    def __init__(self, base_module: nn.Module):
        super(ManifoldMixupModule, self).__init__()
        self._base_module = base_module
        self._mixup = None
        self.is_mixing = False

    def forward(self, input):
        if self.training and self._mixup is not None and self.is_mixing:
            input = self._mixup.transform_input(input)
        return self._base_module(input)


class ManifoldMixupCallback(Callback):
    """
    Implement manifold mixup regularization as a callback. Each iteration, pick a random layer and transform its output
    and label:
    x = tau x_1 + (1 - tau) x_2
    y = tau y_1 + (1 - tau) y_2
    Reference: https://arxiv.org/pdf/1806.05236.pdf
    Based on fastai implementation: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py
    """

    def __init__(self, learner, modules: Sequence[ManifoldMixupModule], alpha: float=2.0):
        for module in modules:
            module._mixup = self

        self._modules = modules
        self._alpha = alpha
        self._learner = learner
        self._old_loss = learner._criterion
        self._old_reduction = getattr(self._old_loss, 'reduction')
        self._shuffle = None
        self._lambd = None

    def on_batch_begin(self, data: Dict[str, Any], train) -> Dict[str, Any]:
        if train:
            self._learner._criterion = self.transform_loss(self._learner._criterion, True)
            labels = data["labels"]
            self._shuffle = torch.randperm(labels.size(0)).to(labels.device)
            self._lambd = self.get_lambd(labels.size(0), labels.device)
            data['labels'] = self.transform_labels(data['labels'])
            mix_ind = np.random.choice(len(self._modules))
            for ind in range(len(self._modules)):
                if mix_ind == ind:
                    self._modules[ind].is_mixing = True
                else:
                    self._modules[ind].is_mixing = False
        return data

    def on_batch_end(self, logs: Dict[str, Any]):
        self._learner._criterion = self.transform_loss(self._learner._criterion, False)

    def on_train_end(self):
        self.deregister()

    def deregister(self):
        for module in self._modules:
            module._mixup = None

    def transform_input(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        inputs_shuffled = inputs[self._shuffle]
        new_images = (
            inputs * self._lambd.view((self._lambd.size(0), 1, 1, 1))
            + inputs_shuffled * (1 - self._lambd).view((self._lambd.size(0), 1, 1, 1))
        )
        return new_images

    def transform_labels(self, labels):
        labels_shuffled = labels[self._shuffle]
        new_labels = torch.cat([
            labels[:, None].float(),
            labels_shuffled[:, None].float(),
            self._lambd[:, None].float()
        ], 1)
        return new_labels

    def get_lambd(self, batch_size, device):
        lambd = np.random.beta(self._alpha, self._alpha, batch_size)
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        return torch.from_numpy(lambd).float().to(device)

    def transform_loss(self, criterion, train):
        if train:
            setattr(criterion, 'reduction', 'none')
            def transformed_loss(outputs, labels):
                loss1, loss2 = criterion(outputs, labels[:, 0].long()), criterion(outputs, labels[:, 1].long())
                return (loss1 * labels[:, 2] + loss2 * (1 - labels[:, 2])).mean()

            return transformed_loss
        else:
            setattr(self._old_loss, 'reduction', self._old_reduction)
            return self._old_loss


