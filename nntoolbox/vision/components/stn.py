import torch
from torch import nn
import torch.nn.functional as F


class SpatialTransformerModule(nn.Module):
    '''
    Implement a spatial transformer module
    Adapt from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    '''

    def __init__(self):
        super(SpatialTransformerModule, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        ) # regress transformation parameters

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        ) # regressor for 3 x 2 affine matrix

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, input):
        inputs = self.localization(input)
        inputs = inputs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(inputs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input.size())
        return F.grid_sample(input, grid)
