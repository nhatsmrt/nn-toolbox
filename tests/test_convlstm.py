from nntoolbox.vision.components import ConvLSTM2dCell, ConvLSTM2d
import torch
from torch import nn
from torch.optim import Adam
import torchvision


class TestConvLSTM2dCell:
    def test_cell(self):
        input_seq = torch.randn(5, 16, 4, 32, 32)
        expected_output = torch.randn(5, 16, 16, 32, 32)

        cell = ConvLSTM2dCell(4, 16, 3)
        optimizer = Adam(cell.parameters())
        loss_fn = nn.MSELoss()
        original_loss = None

        for i in range(10):
            outputs = []
            hc = None

            for t in range(input_seq.shape[0]):
                input = input_seq[0]
                hc = cell(input, hc)
                outputs.append(hc[0])

            output = torch.stack(outputs, dim=0)
            assert output.shape == expected_output.shape

            loss_val = loss_fn(output, expected_output)

            if original_loss is None:
                original_loss = loss_val.item()

            if loss_val.item() < 1e-2:
                return

            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert loss_val.item() < original_loss

    def test_layer(self):
        input_seq = torch.randn(5, 16, 4, 32, 32)
        expected_output = torch.randn(5, 16, 16, 32, 32)

        layer = ConvLSTM2d(4, 16, 3)
        optimizer = Adam(layer.parameters())
        loss_fn = nn.MSELoss()
        original_loss = None

        for i in range(10):
            output, _ = layer(input_seq)
            assert output.shape == expected_output.shape

            loss_val = loss_fn(output, expected_output)

            if original_loss is None:
                original_loss = loss_val.item()

            if loss_val.item() < 1e-2:
                return

            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert loss_val.item() < original_loss
