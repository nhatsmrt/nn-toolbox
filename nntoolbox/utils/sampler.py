from torch.utils.data import Sampler, WeightedRandomSampler, BatchSampler, Dataset
import torch
import numpy as np
from .utils import compute_num_batch


__all__ = ['MultiRandomSampler', 'BatchSampler', 'BatchStratifiedSampler']


class MultiRandomSampler(Sampler):
    """Samples elements randomly to form multiple batches

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, batch_size:int, replacement=False):
        self.data_source = data_source
        self.replacement = replacement
        self.batch_size = batch_size

        assert self.batch_size <= len(data_source)

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        indices = []
        # for _ in range(compute_num_batch(n, self.batch_size)):
        if self.replacement:
            indices = torch.randint(high=n, size=(len(self.data_source),), dtype=torch.int64).tolist()
        else:
            indices = torch.randperm(n).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class BalancedSampler(WeightedRandomSampler):
    """
    For each data point, sample it with weight inversely proportional to the number of points of its class
    """
    def __init__(self, data_source: Dataset, num_samples: int, replacement: bool=True):
        """
        :param data_source: dataset
        :param num_samples: number of samples to draw
        :param replacement: if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row
        """
        all_labels = []
        for inputs, labels in data_source:
            all_labels.append(labels)


        class_weights = 1.0 / np.bincount(all_labels)
        weights = list(map(lambda label: class_weights[label], all_labels))
        # weights = np.array(weights) / np.sum(weights)
        super(BalancedSampler, self).__init__(weights, num_samples, replacement)


class BatchStratifiedSampler(BatchSampler):
    """Ensure that each class in the batch has the same number of examples"""
    def __init__(self, data_source: Dataset, n_sample_per_class: int, n_class_per_batch: int, drop_last: bool=False):
        """
        :param data_source: dataset
        :param batch_size: size of each batch
        :param n_class_per_batch: number of class to sample each batch
        :param drop_last: if ``True``, drop the last batch.
        """
        all_labels = []
        for inputs, labels in data_source:
            all_labels.append(labels)

        self.labels = np.array(all_labels)
        self.label_counts = np.bincount(all_labels)
        assert len(self.label_counts) >= n_class_per_batch

        self.n_data = int(np.sum(all_labels))
        self.batch_size = n_class_per_batch * n_sample_per_class
        self.n_sample_per_class = n_sample_per_class
        self.n_class_per_batch, self.drop_last = n_class_per_batch, drop_last

    def __iter__(self):
        for _ in range(self.__len__()):
            classes = np.random.choice(len(self.label_counts), self.n_class_per_batch, replace=False)
            batch_idx = []
            for c in classes:
                c_idx = np.where(self.labels == c)[0]
                batch_idx.append(np.random.choice(c_idx, size=self.n_sample_per_class, replace=True))

            yield np.concatenate(batch_idx, axis=0)

    def __len__(self):
        if self.drop_last:
            return self.n_data // self.batch_size
        return compute_num_batch(self.n_data, self.batch_size)

