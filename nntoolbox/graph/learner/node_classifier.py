import numpy as np
import torch
from torch import nn
from torch.optim import Adam


class NodeClassificationLearner:
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()


    def learn(
            self, model, input, label,
            indices_train, n_epoch=5000,
            verbose=True
    ):
        model.train()

        input = torch.from_numpy(input).float()
        label = torch.from_numpy(label).long()
        optimizer = Adam(model.parameters(), lr=0.001)

        for e in range(n_epoch):
            output = model.forward(input)[indices_train]
            loss = self._loss(output, label)

            if verbose:
                print("Epoch " + str(e) + " with loss " + str(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
