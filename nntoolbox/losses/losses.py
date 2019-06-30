from torch.nn import Module, MSELoss
from torch import Tensor
from nntoolbox.utils import to_onehot
import torch
import torch.nn.functional as F
from typing import Optional


__all__ = ['RMSELoss', 'SmoothedCrossEntropy']


class RMSELoss(MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', eps: float=1e-8):
        super(RMSELoss, self).__init__(size_average, reduce, reduction)
        self._eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(super().forward(input, target) + self._eps)


# UNTESTED
class SmoothedCrossEntropy(Module):
    '''
    A smooth cross entropy loss. Based on:
    http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
    '''
    def __init__(self, weight: Optional[Tensor]=None, reduction: str='mean', eps: float=0.1):
        assert reduction == 'mean' or reduction =='sum' or reduction == 'none'
        if weight is not None:
            assert len(weight.shape) == 1
        super(SmoothedCrossEntropy, self).__init__()
        self.eps = eps
        self.weight = weight
        self.reduction = reduction

    def forward(self, output: Tensor, label: Tensor) -> Tensor:
        '''
        :param output: Predicted class scores. (batch_size, C, *)
        :param label: The true label. (batch_size, *)
        :return:
        '''
        if self.weight is not None:
            assert len(self.weight) == output.shape[1]

        smoothed_label = self.smooth_label(label, output.shape[1])
        output = F.log_softmax(output, 1)
        loss = -output * smoothed_label
        if self.weight is not None:
            weight_shape = [1, self.weight.shape[0]] + [1 for _ in range(len(output.shape) - 2)]
            weight = self.weight.view(weight_shape)
            loss = loss * weight
        loss = loss.sum(1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    def smooth_label(self, label: Tensor, n_class: int) -> Tensor:
        '''
        Smooth the label
        :param label: (batch_size, *)
        :param n_class: number of class of the output
        :return: (batch_size, C, *)
        '''
        label_oh = to_onehot(label, n_class).float()
        return (1 - self.eps) * label_oh + self.eps / (n_class - 1) * (1 - label_oh)
