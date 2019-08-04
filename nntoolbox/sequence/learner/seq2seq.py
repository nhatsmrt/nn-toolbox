import torch
from torch import nn
from torch.optim import Adam, Optimizer
from nntoolbox.utils import compute_num_batch
from nntoolbox.sequence.utils import create_mask, get_lengths
from nntoolbox.callbacks import CallbackHandler, Callback
from nntoolbox.metrics import Metric
import random
import numpy as np
from tqdm import trange
from typing import Optional, Iterable, Dict
from ..models import Encoder, Decoder


class Seq2SeqLearnerV2:
    def __init__(
            self, train_iterator, val_iterator, encoder: Encoder, decoder: Decoder,
            criterion: Optional[nn.Module], optimizer: Optimizer, teacher_forcing_ratio: float=1.0,
            pad_token: int=0, SOS_token: int=1, EOS_token: int=2
    ):
        self._models = [encoder, decoder]
        self._train_iterator, self._val_iterator = train_iterator, val_iterator
        self._criterion, self._optimizer = criterion, optimizer
        self._teacher_forcing_ratio = teacher_forcing_ratio
        self._pad_token, self._SOS_token, self._EOS_token = pad_token, SOS_token, EOS_token

    def learn(
            self, n_epoch: int, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, final_metric: str='accuracy'
    ) -> float:
        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()

        for e in range(n_epoch):
            for model in self._models: model.train()
            self._cb_handler.on_epoch_begin()

            # train step

            stop_training = self.evaluate()
            if stop_training:
                break

        return self._cb_handler.on_train_end()

    def evaluate(self) -> bool:
        return False


class Seq2SeqLearner:
    """
    INCOMPLETE
    """
    def __init__(
            self, encoder: Encoder, decoder: Decoder,
            X, Y, X_val, Y_val, device,
            teacher_forcing_ratio=1.0,
            pad_token=0, SOS_token=1, EOS_token=2
    ):
        self._encoder = encoder
        self._decoder = decoder
        self._X = X
        self._Y = Y
        self._X_val = X_val
        self._Y_val = Y_val

        self._encoder_optimizer = Adam(self._encoder.parameters())
        self._decoder_optimizer = Adam(self._decoder.parameters())

        self._teacher_forcing_ratio = teacher_forcing_ratio
        self._loss = nn.CrossEntropyLoss(ignore_index=pad_token)
        self._pad_token = pad_token
        self._SOS_token = SOS_token
        self._EOS_token = EOS_token
        self._device = device

    def learn(self, n_epoch, batch_size, print_every, eval_every):
        """
        :param n_epoch: number of epoch to train
        :param batch_size: size of each batch
        :param print_every
        :param eval_every
        :return:
        """
        indices = np.arange(self._X.shape[1])
        n_batch = compute_num_batch(self._X.shape[1], batch_size)

        mask_X, lengths_X, X = self.prepare_input(self._X)
        mask_Y, lengths_Y, Y = self.prepare_input(self._Y)

        mask_X_val, lengths_X_val, X_val = self.prepare_input(self._X_val)
        mask_Y_val, lengths_Y_val, Y_val = self.prepare_input(self._Y_val)

        iter_cnt = 0
        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._encoder.train()
            self._decoder.train()

            np.random.shuffle(indices)
            for i in trange(n_batch):
                idx = indices[i * batch_size:(i + 1) * batch_size]
                X_batch = X[:, idx]
                Y_batch = Y[:, idx]
                mask_X_batch = mask_X[:, idx]
                lengths_X_batch = lengths_X[idx]


                loss = self.learn_one_iter(
                    X_batch, Y_batch,
                    mask_X_batch,
                    lengths_X_batch
                )

                if i % print_every == 0:
                    print()
                    print(loss)

                iter_cnt += 1

            if e % eval_every == 0:
                self.evaluate(X_val, Y_val, mask_X_val, lengths_X_val, mask_Y_val)

    @torch.no_grad()
    def evaluate(self, X_val, Y_val, mask_X_val, lengths_X_val, mask_Y_val):
        """
        :param X_val:
        :param Y_val:
        :param mask_X_val:
        :param lengths_X_val:
        :return:
        """
        self._encoder.eval()
        self._decoder.eval()

        # use_teacher_forcing = True if random.random() < self._teacher_forcing_ratio else False
        use_teacher_forcing = False

        batch_size = X_val.shape[1]
        hidden = self._encoder.init_hidden(batch_size)
        enc_outputs, hidden = self._encoder(X_val, hidden)

        hidden = enc_outputs.gather(
            dim=0,
            index=(lengths_X_val - 1).view(1, -1).unsqueeze(-1).repeat(1, 1, enc_outputs.shape[2])
        )
        outputs = []
        if use_teacher_forcing:
            self._decoder_input = torch.cat(
                (
                    torch.from_numpy(np.array([[self._SOS_token for _ in range(batch_size)]])).to(self._device),
                    Y_val[:Y_val.shape[0] - 1]
                ),
                dim=0
            )
            outputs, hidden = self._decoder(self._decoder_input, hidden, enc_outputs, mask=mask_X_val)
            outputs = outputs.permute(1, 2, 0)
        else:
            self._decoder_input = torch.from_numpy(np.array([[self._SOS_token for _ in range(batch_size)]])).to(self._device)
            for t in range(X_val.shape[0]):
                output, hidden = self._decoder(self._decoder_input, hidden, enc_outputs, mask=mask_X_val)
                outputs.append(output)
                self._decoder_input = torch.argmax(output, dim=-1)
            outputs = torch.cat(outputs, dim=0).permute(1, 2, 0)

        loss = self._loss(outputs, Y_val.permute(1, 0))

        print()
        print("A random example (output, Y, X):")
        random_ind = np.random.choice(outputs.shape[0])
        print(outputs[random_ind].argmax(dim=0))
        print(Y_val.permute(1, 0)[random_ind])
        print(X_val.permute(1, 0)[random_ind])
        mask_Y_val = mask_Y_val.float().permute(1, 0)
        acc = torch.sum((outputs.argmax(dim=1) == Y_val.permute(1, 0)).float() * mask_Y_val) / torch.sum(mask_Y_val)
        print("Val acc: " + str(acc.item()))
        print("Val loss: " + str(loss.item()))
        print()

    def learn_one_iter(self, X_batch, Y_batch, mask_X_batch, lengths_X_batch):
        self._encoder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()

        use_teacher_forcing = True if random.random() < self._teacher_forcing_ratio else False

        batch_size = X_batch.shape[1]
        hidden = self._encoder.init_hidden(batch_size)
        enc_outputs, hidden = self._encoder(X_batch, hidden)


        hidden = enc_outputs.gather(
            dim=0,
            index=(lengths_X_batch - 1).view(1, -1).unsqueeze(-1).repeat(1, 1, enc_outputs.shape[2])
        )
        outputs = []
        if use_teacher_forcing:
            self._decoder_input = torch.cat(
                (
                    torch.from_numpy(np.array([[self._SOS_token for _ in range(batch_size)]])).to(self._device),
                    Y_batch[:Y_batch.shape[0] - 1]
                ),
                dim=0
            )
            outputs, hidden = self._decoder(self._decoder_input, hidden, enc_outputs, mask=mask_X_batch)
            outputs = outputs.permute(1, 2, 0)
        else:
            self._decoder_input = torch.from_numpy(np.array([[self._SOS_token for _ in range(batch_size)]])).to(self._device)
            for t in range(X_batch.shape[0]):
                output, hidden = self._decoder(self._decoder_input, hidden, enc_outputs, mask=mask_X_batch)
                outputs.append(output)
                # self._decoder_input = torch.argmax(output, dim=-1)
                self._decoder_input = Y_batch[t:t+1]
            outputs = torch.cat(outputs, dim=0).permute(1, 2, 0)

        loss = self._loss(outputs, Y_batch.permute(1, 0))
        loss.backward()

        self._encoder_optimizer.step()
        self._decoder_optimizer.step()

        return loss.item()

    def prepare_input(self, X):
        """
        :param X: sequence length of shape (seq_length, batch_size)
        :return: mask and lengths
        """
        mask = create_mask(X, self._pad_token)

        lengths = get_lengths(mask)
        outputs = (
            torch.from_numpy(mask).to(self._device),
            torch.from_numpy(lengths).long().to(self._device),
            torch.from_numpy(X).long().to(self._device)
        )
        return outputs
