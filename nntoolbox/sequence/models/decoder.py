import torch
from torch import nn
from ..components import AdditiveAttention


class Decoder(nn.Module):

    def __init__(self, output_size, hidden_size, embedding_dim, max_length, enc_dim, device, dropout_p=0.1):
        super(Decoder, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._enc_dim = enc_dim
        self._output_size = output_size
        self._max_length = max_length
        self._device = device

        self._embedding = nn.Embedding(output_size, embedding_dim)
        self._dropout = nn.Dropout(dropout_p)


    def forward(self, input, hidden, mask=None, encoder_outputs=None):
        raise NotImplementedError

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self._hidden_size, device=self._device)




class AttentionalDecoder(Decoder):
    def __init__(self, hidden_size, output_size, embedding_dim, max_length, enc_dim, device, dropout_p=0.1):
        '''
        :param embedding_dim: dimension of
        :param hidden_size:
        :param output_size:
        :param max_length:
        :param device:
        :param dropout_p:
        '''
        super(AttentionalDecoder, self).__init__(output_size, hidden_size, embedding_dim, max_length, enc_dim, device, dropout_p)
        self._gru = nn.GRU(
            input_size=self._embedding_dim + self._enc_dim,
            hidden_size=self._hidden_size
        )
        self._attention = AdditiveAttention(
            input_dim=self._enc_dim,
            query_dim=embedding_dim,
            hidden_dim=128,
            max_length=max_length,
            return_summary=True
        )
        self._op = nn.Sequential(
            nn.Linear(self._hidden_size, self._output_size),
            nn.Softmax(dim=-1)
        )
        self.to(device)


    def forward(self, input, hidden, encoder_outputs=None, mask=None):
        '''
        :param input: current time step input: (seq_len, n_batch, 1)
        :param hidden: hidden state of decoder's previous timestep: (seq_len, n_batch, emb_dim)
        :param encoder_outputs: outputs of encoder: (max_length, n_batch, enc_op_dim)
        :return: output
        '''
        embedded = self._dropout(self._embedding(input)) # (seq_length=1, n_batch, embedding_dim)
        encoder_outputs_sum, mask = self._attention(
            inputs=encoder_outputs,
            query=embedded[0],
            mask=mask
        ) # (n_batch, enc_op_dim)
        concat_input = torch.cat(
            (embedded, encoder_outputs_sum.unsqueeze(0)),
            dim=-1
        ) # (1, n_batch, embedding_dim + enc_op_dim)

        output, hidden = self._gru(concat_input, hidden)
        op = self._op(output)
        return op, hidden




