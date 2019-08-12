import torch
from torch import nn, Tensor
from typing import Tuple
from ..components import ResidualRNN


__all__ = ['Encoder', 'RNNEncoder', 'GRUEncoder']


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers, bidirectional, device, pad_token=0, drop_rate=0.1):
        super(Encoder, self).__init__()
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._device = device

        self._embedding = nn.Embedding(input_size, self._embedding_dim, padding_idx=pad_token)
        self._dropout = nn.Dropout(drop_rate)

    def forward(self, input: Tensor, states: Tuple[Tensor, ...]) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        :param input: (seq_len, batch_size, input_dim)
        :param states: internal states of the RNN, each having dimension
        (num_layers * num_directions, batch_size, hidden_size)
        :return:

            output: (seq_len, batch, num_directions * hidden_size)

            states: states at final time step, each having dimension
                (num_layers * num_directions, batch_size, hidden_size)
        """
        raise NotImplementedError

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Initialize the first zero hidden state

        :param batch_size:
        :return: Initial internal states, each of dim (num_layers * num_directions, batch_size, hidden_size)
        """
        raise NotImplementedError


class RNNEncoder(Encoder):
    def __init__(
            self, rnn, input_size, hidden_size, embedding_dim,
            num_layers, bidirectional, device, pad_token=0, drop_rate=0.1
    ):
        super(RNNEncoder, self).__init__(
            input_size, hidden_size, embedding_dim, num_layers, bidirectional, device, pad_token, drop_rate
        )
        self.rnn = rnn

    def forward(self, input: Tensor, states: Tuple[Tensor, ...]) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        embedded = self._dropout(self._embedding(input))
        output, hidden = self.rnn(embedded, states)
        return output, hidden


class GRUEncoder(RNNEncoder):
    def __init__(
            self, input_size, hidden_size, embedding_dim, device, bias=False,
            num_layers=1, dropout=0, bidirectional=False, pad_token=0, drop_rate=0.1
    ):
        super(GRUEncoder, self).__init__(
            nn.GRU(
                embedding_dim, hidden_size,
                bias=bias, num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            ),
            input_size, hidden_size, embedding_dim, num_layers, bidirectional, device, pad_token, drop_rate
        )

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Initialize the first zero hidden state

        :param batch_size:
        :return: Initial hidden state, of dimensision (num_layers * num_directions, batch_size, hidden_size)
        """
        first_dim = self._num_layers
        if self._bidirectional:
            first_dim *= 2
        return (torch.zeros(first_dim, batch_size, self._hidden_size, device=self._device),)


class LSTMEncoder(RNNEncoder):
    def __init__(
            self, input_size, hidden_size, embedding_dim, device, bias=False,
            num_layers=1, dropout=0, bidirectional=False, pad_token=0, drop_rate=0.1
    ):
        super(LSTMEncoder, self).__init__(
            nn.LSTM(
                embedding_dim, hidden_size,
                bias=bias, num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            ),
            input_size, hidden_size, embedding_dim, num_layers, bidirectional, device, pad_token, drop_rate
        )

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Initialize the first zero hidden state

        :param batch_size:
        :return: Initial hidden state and cell state, each of dim (num_layers * num_directions, batch_size, hidden_size)
        """
        first_dim = self._num_layers
        if self._bidirectional:
            first_dim *= 2
        return (
            torch.zeros(first_dim, batch_size, self._hidden_size, device=self._device),
            torch.zeros(first_dim, batch_size, self._hidden_size, device=self._device)
        )


class ResidualRNNEncoder(RNNEncoder):
    def __init__(
            self, base_rnn, input_size, hidden_size, embedding_dim, device, bias=False,
            num_layers=1, dropout=0, bidirectional=False, pad_token=0, drop_rate=0.1
    ):
        super(ResidualRNNEncoder, self).__init__(
                ResidualRNN(
                    base_rnn=base_rnn, input_size=embedding_dim,
                    bias=bias, num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional
                ),
                input_size, hidden_size, embedding_dim, num_layers, bidirectional, device, pad_token, drop_rate
        )


# class ResidualGRUEncoder(RNNEncoder, GRUEncoder):
#     def __init__(
#             self, input_size, hidden_size, embedding_dim, device, bias=False,
#             num_layers=1, dropout=0, bidirectional=False, pad_token=0, drop_rate=0.1
#     ):
#         super(ResidualGRUEncoder, self).__init__(
#             nn.GRU, input_size, hidden_size, embedding_dim, num_layers, bidirectional,
#             device, bias, num_layers, dropout, bidirectional, pad_token, drop_rate
#         )

#
# class GRUEncoder(Encoder):
#     def __init__(
#             self, input_size, hidden_size, embedding_dim, device, bias=False,
#             num_layers=1, dropout=0, bidirectional=False, pad_token=0, drop_rate=0.1):
#         super(GRUEncoder, self).__init__(
#             input_size, hidden_size, embedding_dim,
#             num_layers, bidirectional, device, pad_token, drop_rate)
#         self._gru = nn.GRU(
#             embedding_dim, hidden_size,
#             bias=bias, num_layers=num_layers,
#             dropout=dropout,
#             bidirectional=bidirectional
#         )
#         # self._gru = ResidualRNN(
#         #     nn.GRU, input_size=hidden_size,
#         #     bias=bias, num_layers=num_layers,
#         #     dropout=dropout,
#         # )
#         self.to(device)
#
#     def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
#         embedded = self._dropout(self._embedding(input))
#         output, hidden = self._gru(embedded, hidden)
#         return output, hidden
#
