from torch.utils.data import Sampler
import torch
from .utils import compute_num_batch


__all__ = ['MultiRandomSampler']


class MultiRandomSampler(Sampler):
    r"""Samples elements randomly to form multiple batches

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
