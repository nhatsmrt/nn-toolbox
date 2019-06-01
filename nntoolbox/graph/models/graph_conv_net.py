import numpy as np
import torch
from torch import nn
from ..components import GraphConvolutionLayer
from ..utils import compute_normed_laplacian

class GraphConvNet(nn.Sequential):
    def __init__(self, in_features, n_class, G):
        super(GraphConvNet, self).__init__()
        normed_laplacian = torch.from_numpy(compute_normed_laplacian(G)).float()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Dropout(),
                GraphConvolutionLayer(
                    in_features = in_features,
                    out_features = 256,
                    normed_laplacian = normed_laplacian,
                    loss = nn.ReLU
                ),
                nn.Dropout(),
                GraphConvolutionLayer(
                    in_features = 256,
                    out_features = n_class,
                    normed_laplacian = normed_laplacian
                )
            )
        )

    def predict(self, input):
        self.eval()
        op = self.forward(torch.from_numpy(input).float()).detach().numpy()
        return np.argmax(op, axis = -1)
