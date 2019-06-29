import torch
import numpy as np
import copy
from torch.nn import Module


def compute_num_batch(data_size: int, batch_size: int):
    '''
    Compute number of batches per epoch
    :param data_size: number of datapoints
    :param batch_size: number of datapoints per batch
    :return:
    '''
    return int(np.ceil(data_size / float(batch_size)))


def copy_model(model: Module):
    '''
    Return an exact copy of the model (both architecture and initial weights, without tying the weights)
    :param model: model to be copied
    :return: a copy of the model
    '''
    return copy.deepcopy(model)


def save_model(model: Module, path: str):
    '''
    :param model:
    :param path: path to save model at
    '''
    torch.save(model.state_dict(), path)
    print("Model saved")


def load_model(model: Module, path: str):
    '''
    Load the model from path
    :param model
    :param path: path of saved model
    '''
    model.load_state_dict(torch.load(path))
    print("Model loaded")


def get_device():
    '''
    :return: a torch device object (gpu if exists)
    '''
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_trainable_parameters(model: Module):
    return filter(lambda p: p.requires_grad, model.parameters())
