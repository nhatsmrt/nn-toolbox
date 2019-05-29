from torch import nn
from torch.nn import functional as F

class L2NormalizationLayer(nn.Module):
    def __init__(self):
        super(L2NormalizationLayer, self).__init__()

    def forward(self, input):
        return F.normalize(input, dim=-1, p=2)
