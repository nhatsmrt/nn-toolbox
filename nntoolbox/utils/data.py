from torch.utils.data import Dataset
from .utils import get_device
import torch
from numpy import ndarray
from torch import float32, long, Tensor
import pandas as pd
from typing import Optional, List, Iterable, Union
from torch.utils.data import DataLoader
from torchtext.data import Iterator


__all__ = ['SupervisedDataset', 'get_first_batch', 'grab_next_batch']


class SupervisedDataset(Dataset):
    def __init__(self, inputs: ndarray, labels: ndarray, device=get_device(), transform=None):
        assert len(inputs) == len(labels)
        self._device = device
        self.inputs = torch.from_numpy(inputs)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    @classmethod
    def from_csv(cls, path: str, label_name: str, data_fields: Optional[List[str]]=None, device=get_device()):
        """Create a supervised dataset from csv file"""
        assert path.endswith(".csv")
        df = pd.read_csv(path)
        labels = df[label_name].values
        if data_fields is None:
            inputs = df.drop(label_name, axis=1).values
        else:
            inputs = df[data_fields].values
        return cls(inputs, labels, device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        input, label = self.prepare_arr(self.inputs[index], float32), self.prepare_arr(self.labels[index], long)
        if self.transform is not None:
            input = self.transform(input)
        return input, label

    def prepare_arr(self, tensor: Tensor, dtype):
        return tensor.to(dtype).to(self._device)


def get_first_batch(data: DataLoader, callbacks: Optional[Iterable['Callback']]=None):
    """
    Get the first batch from dataloader

    :param data: the dataloader
    :param callbacks: the list of callbacks to applied to data
    """
    first_batch = next(iter(data))
    if callbacks is None or len(callbacks) == 0:
        return first_batch
    else:
        if isinstance(first_batch, tuple):
            data = {"inputs": first_batch[0], "labels": first_batch[1]}
        else:
            data = {"inputs": first_batch[0], "labels": first_batch[1]}
        for callback in callbacks:
            data = callback.on_batch_begin(data, True)
        return data["inputs"] if callbacks is None else data["inputs"], data["labels"]


def grab_next_batch(data: Union[DataLoader, Iterator]):
    """Grab the next batch from dataloader"""
    return next(iter(data))
