import torch
from torch.optim import Optimizer

def change_lr(optim:Optimizer, lr:float):
    for param_group in optim.param_groups:
        param_group['lr'] = lr