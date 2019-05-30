import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers, bidirectional, device):
        super(Encoder, self).__init__()
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._device = device

        self._embedding = nn.Embedding(input_size, self._embedding_dim)

    def forward(self, input, hidden):
        raise NotImplementedError

    def init_hidden(self, batch_size):
        first_dim = self._num_layers
        if self._bidirectional:
            first_dim *= 2
        return torch.zeros(first_dim, batch_size, self._hidden_size, device=self._device)


class GRUEncoder(Encoder):
    def __init__(self, input_size, hidden_size, embedding_dim, device, bias=False, num_layers=1, dropout=0, bidirectional=False):
        super(GRUEncoder, self).__init__(input_size, hidden_size, embedding_dim, num_layers, bidirectional, device)
        self._gru = nn.GRU(
            embedding_dim, hidden_size,
            bias=bias, num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, input, hidden):
        embedded = self._embedding(input)
        output, hidden = self._gru(embedded, hidden)
        return output, hidden

