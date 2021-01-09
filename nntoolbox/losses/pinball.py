from torch import nn, Tensor
import torch


__all__ = ['PinballLoss']


class PinballLoss(nn.Module):
    """
    Pinball loss for quantile regression:

        L_tau(y_true, y_pred) = max(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))

    References:

        https://www.tensorflow.org/addons/api_docs/python/tfa/losses/PinballLoss

        Ingo Steinwart and Andreas Christmann, "Estimating conditional quantiles with the help of the pinball loss."
        https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
    """
    def __init__(self, tau: float=0.5, reduction: str='mean'):
        super().__init__()
        assert 0.0 < tau < 1.0
        assert reduction in ['mean', 'sum', 'none']

        self.tau, self.reduction = tau, reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        error = target - input
        losses = torch.stack([self.tau * error, (self.tau - 1.0) * error], dim=0).max(dim=0)[0]

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
