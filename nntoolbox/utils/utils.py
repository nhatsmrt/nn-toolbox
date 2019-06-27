import torch
import numpy as np
import copy
from torch.nn import Module
from torch import Tensor


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


def compute_gradient(output: Tensor, model: Module):
    ret = []
    output.backward(retain_graph=True)
    for parameter in model.parameters():
        ret.append(parameter.grad)
        parameter.grad = None # Reset gradient accumulation
    return ret


def update_gradient(gradients, model: Module, fn=lambda x:x):
    for gradient, parameter in zip(gradients, model.parameters()):
        parameter.grad = fn(gradient) # Reset gradient accumulation


def accumulate_gradient(gradients, model, fn=lambda x:x):
    for gradient, parameter in zip(gradients, model.parameters()):
        parameter.grad += fn(gradient) # Reset gradient accumulation


def compute_gradient_norm(output: Tensor, model: Module):
    '''
    Compute the norm of the gradient of an output (e.g a loss) with respect to a model parameters
    :param output:
    :param model:
    :return:
    '''
    ret = 0
    output.backward(retain_graph=True)
    for parameter in model.parameters():
        grad = parameter.grad
        ret += grad.pow(2).sum().cpu().detach().numpy()
        parameter.grad = None # Reset gradient accumulation

    return ret


def get_trainable_parameters(model: Module):
    return filter(lambda p: p.requires_grad, model.parameters())
