import torch
from torch import nn
from torch.optim import Adam, SGD
from nntoolbox.utils import compute_num_batch
from nntoolbox.sequence.utils import create_mask, get_lengths
import random
import numpy as np


class Seq2SeqLearner:
    def __init__(self, teacher_forcing_ratio=1.0, pad_token=0, SOS_token=1, EOS_token=2):
        self._teacher_forcing_ratio = teacher_forcing_ratio
        self._loss = nn.CrossEntropyLoss(ignore_index=pad_token)
        self._pad_token = pad_token
        self._SOS_token = SOS_token
        self._EOS_token = EOS_token


    def learn(self, encoder, decoder, X, Y, n_epoch, batch_size):
        '''
        :param encoder:
        :param decoder:
        :param X:
        :param Y:
        :param n_epoch: number of epoch to train
        :param batch_size: size of each batch
        :return:
        '''

        encoder_optimizer = Adam(encoder.parameters())
        decoder_optimizer = Adam(decoder.parameters())
        indices = np.arange(X.shape[1])
        n_batch = compute_num_batch(X.shape[1], batch_size)

        mask_X, lengths_X, X = self.prepare_input(X)
        mask_Y, lengths_Y, Y = self.prepare_input(Y)

        for e in range(n_epoch):
            print("Epoch " + str(e))
            encoder.train()
            decoder.train()

            np.random.shuffle(indices)
            for i in range(n_batch):
                idx = indices[i * batch_size:(i + 1) * batch_size]
                X_batch = X[:, idx]
                Y_batch = Y[:, idx]
                mask_X_batch = mask_X[:, idx]
                lengths_X_batch = lengths_X[idx]


                self.learn_one_iter(
                    encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    X_batch, Y_batch,
                    mask_X_batch,
                    lengths_X_batch
                )


    def learn_one_iter(self, encoder, decoder, encoder_optimizer, decoder_optimizer, X_batch, Y_batch, mask_X_batch, lengths_X_batch):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        use_teacher_forcing = True if random.random() < self._teacher_forcing_ratio else False

        batch_size = X_batch.shape[1]
        hidden = encoder.init_hidden(batch_size)
        enc_outputs, hidden = encoder(X_batch, hidden)


        decoder_input = torch.from_numpy(np.array([[self._SOS_token for _ in range(batch_size)]]))
        hidden = enc_outputs.index_select(dim=0, index=lengths_X_batch - 1)
        # hidden = hidden.permute((1, 0, 2)).contiguous().view((hidden.shape[1], -1)).unsqueeze(0)
        outputs = []
        for t in range(X_batch.shape[0]):
            output, hidden = decoder(decoder_input, hidden, enc_outputs, mask=mask_X_batch)
            outputs.append(output)
            if use_teacher_forcing:
                decoder_input = Y_batch[t:t+1]
            else:
                decoder_input = torch.argmax(output, dim=-1)



        outputs = torch.cat(outputs, dim=0).permute(1, 2, 0)
        loss = self._loss(outputs, Y_batch.permute(1, 0))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        print(loss)

        return loss


    def prepare_input(self, X):
        '''
        :param X: sequence length of shape (seq_length, batch_size)
        :return: mask and lengths
        '''
        mask = create_mask(X, self._pad_token)

        lengths = get_lengths(mask)

        return mask, torch.from_numpy(lengths).long(), torch.from_numpy(X).long()
