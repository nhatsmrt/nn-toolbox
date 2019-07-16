from torch import Tensor
from torch.nn import Module
from torch.nn.init import normal_, constant_


def normal_init(module: Module, mean: float, std: float):
    """
    Initialize the weight of a module to normal distribution of given mean and std
    If module has bias, assign it to zero constant
    (UNTESTED)

    :param module: must have weight tensor
    :param: mean of distribution
    :param: std: standard deviation of the distribution
    """
    normal_(module.weight.data, mean=mean, std=std)
    if module.bias is not None:
        constant_(module.bias.data, val=0.0)
