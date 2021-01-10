from nntoolbox.components import Snake
import torch
from torch import nn, optim
from math import pi


class TestSnake:
    def test_snake_extrapolate(self):
        inputs = (torch.rand((1000, 1)) - 0.5) / 0.5 * pi
        targets = torch.sin(inputs)

        model = nn.Sequential(nn.Linear(1, 512), Snake(), nn.Linear(512, 1))
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()

        for i in range(15000):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()


        test_inputs = (torch.rand((1000, 1)) - 0.5) / 0.5 * pi + 100
        test_targets = torch.sin(test_inputs)

        assert abs(loss_fn(model(test_inputs), test_targets)) < 1e-3