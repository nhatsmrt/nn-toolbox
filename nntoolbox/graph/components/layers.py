import torch
from torch import nn

class GraphConvolutionLayer(nn.Sequential):
    def __init__(self, in_features, out_features, normed_laplacian, loss = None):
        super(GraphConvolutionLayer, self).__init__()
        self.add_module(
            "main",
            nn.Linear(in_features = in_features, out_features = out_features),
        )

        if loss is not None:
            self.add_module(
                "loss",
                loss()
            )

        self._normed_laplacian = normed_laplacian

    def forward(self, input):
        return super(GraphConvolutionLayer, self).forward(torch.mm(self._normed_laplacian, input))
