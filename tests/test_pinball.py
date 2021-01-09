import numpy as np
from nntoolbox.losses import PinballLoss
import torch


class TestPinball:
    def test_pinball(self):
        """
        Adopt from https://www.tensorflow.org/addons/api_docs/python/tfa/losses/PinballLoss
        """
        target = torch.from_numpy(np.array([0., 0., 1., 1.]))
        input = torch.from_numpy(np.array([1., 1., 1., 0.]))

        loss = PinballLoss(tau=0.1)
        assert abs(loss(input, target).item() - 0.475) < 1e-3