import torch
import numpy as np
import copy

def compute_num_batch(data_size, batch_size):
    '''
    Compute number of batches per epoch
    :param data_size: number of datapoints
    :param batch_size: number of datapoints per batch
    :return:
    '''
    return int(np.ceil(data_size / float(batch_size)))

def copy_model(model):
    '''
    Return an exact copy of the model (both architecture and initial weights, without tying the weights)
    :param model: model to be copied
    :return: a copy of the model
    '''
    return copy.deepcopy(model)

def save_model(model, PATH):
    '''
    :param model:
    :param PATH: path to save model at
    '''
    torch.save(model.state_dict(), PATH)
    print("Model saved")


def load_model(model, PATH):
    '''
    Load the model from path
    :param model
    :param PATH: path of saved model
    '''
    model.load_state_dict(torch.load(PATH))
    print("Model loaded")
