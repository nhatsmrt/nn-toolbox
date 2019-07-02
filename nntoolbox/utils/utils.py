import torch
import numpy as np
import copy
from torch.nn import Module
from torch import Tensor
from typing import Optional, List


def compute_num_batch(data_size: int, batch_size: int):
    """
    Compute number of batches per epoch
    :param data_size: number of datapoints
    :param batch_size: number of datapoints per batch
    :return:
    """
    return int(np.ceil(data_size / float(batch_size)))


def copy_model(model: Module) -> Module:
    """
    Return an exact copy of the model (both architecture and initial weights, without tying the weights)
    :param model: model to be copied
    :return: a copy of the model
    """
    return copy.deepcopy(model)


def save_model(model: Module, path: str):
    """
    :param model:
    :param path: path to save model at
    """
    torch.save(model.state_dict(), path)
    print("Model saved")


def load_model(model: Module, path: str):
    """
    Load the model from path
    :param model
    :param path: path of saved model
    """
    model.load_state_dict(torch.load(path))
    print("Model loaded")


def get_device():
    """
    :return: a torch device object (gpu if exists)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_trainable_parameters(model: Module) -> List[Tensor]:
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def count_trainable_parameters(model: Module) -> int:
    """
    Based on https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_onehot(label: Tensor, n_class: Optional[int]=None) -> Tensor:
    """
    Return one hot encoding of label
    :param label:
    :param n_class:
    :return:
    """
    if n_class is None:
        n_class = torch.max(label) + 1
    label_oh = torch.zeros([label.shape[0], n_class] + list(label.shape)[1:]).long().to(label.device)
    label = label.unsqueeze(1)
    label_oh.scatter_(dim=1, index=label, value=1)
    return label_oh

