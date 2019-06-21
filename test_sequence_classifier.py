from torch import nn
from torchtext import data
from torchtext.datasets import IMDB, SST
from nntoolbox.utils import get_device, get_trainable_parameters
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from nntoolbox.sequence.learner import SequenceClassifierLearner
from nntoolbox.sequence.components import AdditiveContextEmbedding, AdditiveAttention, ResidualRNN
from nntoolbox.components import MLP
from functools import partial


MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 32

TEXT = data.Field(tokenize='spacy', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)
train_data, val_data, test_data = SST.splits(
    text_field=TEXT,
    label_field=LABEL

)

train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=get_device()
)

TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
OUTPUT_DIM = 3
NUM_LAYERS = 2
BIDIRECTIONAL = True
PADDING_IDX = TEXT.vocab.stoi[TEXT.pad_token]


class SequenceFeatureExtractor(nn.Module):
    def __init__(self, pool):
        super(SequenceFeatureExtractor, self).__init__()
        self._pool = pool()

    def forward(self, input, sequence_lengths):
        '''
        :param input: (seq_length, batch_size, n_features)
        :param sequence_lengths: (batch size)
        :return: (batch_size, n_features)
        '''
        batch_size = len(sequence_lengths)
        features = [self._pool(input[:sequence_lengths[i], i:i + 1, :]) for i in range(batch_size)]
        return torch.cat(features, dim=0)


class AveragePool(nn.Module):
    def __init__(self, dim):
        super(AveragePool, self).__init__()
        self._dim = dim

    def forward(self, input):
        return torch.mean(input, dim=self._dim)


class MaxPool(nn.Module):
    def __init__(self, dim):
        super(MaxPool, self).__init__()
        self._dim = dim

    def forward(self, input):
        return torch.max(input, dim=self._dim).values()


class ConcatPool(nn.Module):
    def __init__(self, pool_dim, concat_dim):
        super(ConcatPool, self).__init__()
        self._pool_dim = pool_dim
        self._concat_dim = concat_dim - 1 if pool_dim < concat_dim and concat_dim > 0 else concat_dim

    def forward(self, input):
        max = torch.max(input, dim=self._pool_dim).values
        avg = torch.mean(input, dim=self._pool_dim)
        return torch.cat([max, avg], dim=self._concat_dim)


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, output_dim, num_layers, bidirectional, padding_idx):
        super(RNNClassifier, self).__init__()
        self._embedding = AdditiveContextEmbedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        # self._rnn = nn.GRU(
        #     input_size=embedding_dim, hidden_size=hidden_size,
        #     num_layers=num_layers, bidirectional=bidirectional,
        #     dropout=0.5
        # )
        # in_features = hidden_size * 2 if bidirectional else hidden_size
        # self._op = MLP(in_features=in_features * 2, out_features=output_dim)
        self._rnn = ResidualRNN(
            nn.GRU, input_size=embedding_dim, num_layers=num_layers,
            bidirectional=bidirectional, dropout=0.5
        )
        in_features = embedding_dim * 2 if bidirectional else hidden_size
        self._op = MLP(in_features=in_features * 2, out_features=output_dim)

        self._dropout = nn.Dropout()

        self._embedding.weight.data.copy_(TEXT.vocab.vectors)
        self._embedding.weight.data[padding_idx] = torch.zeros(embedding_dim)
        self._feature_extactor = SequenceFeatureExtractor(partial(ConcatPool, 0, -1))

    def forward(self, input, text_lengths):
        embedded = self._dropout(self._embedding(input))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, hidden = self._rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = output.gather(
        #     dim=0,
        #     index=(text_lengths - 1).view(1, -1).unsqueeze(-1).repeat(1, 1, output.shape[2])
        # ).squeeze(0)
        output_features = self._feature_extactor(output, output_lengths)
        return self._op(output_features)


model = RNNClassifier(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    bidirectional=BIDIRECTIONAL,
    padding_idx=PADDING_IDX
)

optimizer = torch.optim.Adam(params=get_trainable_parameters(model))
learner = SequenceClassifierLearner(train_iterator, val_iterator, model, nn.CrossEntropyLoss(), optimizer=optimizer)
callbacks = [
    Tensorboard(),
    ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=10),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
    # EarlyStoppingCB(monitor='accuracy', mode='max', patience=15)
]
metrics = {
    "accuracy": Accuracy(),
    "loss": Loss()
}
final = learner.learn(
    n_epoch=500,
    callbacks=callbacks,
    metrics=metrics,
    final_metric='accuracy'
)