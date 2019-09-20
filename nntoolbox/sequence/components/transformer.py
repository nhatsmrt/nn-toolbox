"""Some extra components for transformers model"""
from torch import nn, Tensor
import warnings


class ProductKeyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        warnings.warn("Incomplete!")
