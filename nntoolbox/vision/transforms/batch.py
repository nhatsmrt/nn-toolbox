"""Transform input by batch"""
from torch import Tensor, flip, rot90
from functools import partial
from typing import List, Callable


__all__ = [
    'BatchCompose', 'Identity', 'LambdaTransform',
    'HorizontalFlip', 'VerticalFlip', 'Rotation90',
    'Rotation180', 'Rotation270'
]


hflip = partial(flip, dims=(-1,))
vflip = partial(flip, dims=(-2,))


rotimg = partial(rot90, dims=(-2, -1))
rot90deg = partial(rotimg, k=1)
rot180deg = partial(rotimg, k=2)
rot270deg = partial(rotimg, k=3)


class BatchCompose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, input: Tensor) -> Tensor:
        for transform in self.transforms: input = transform(input)
        return input


class Identity:
    def __call__(self, input: Tensor) -> Tensor: return input


class LambdaTransform:
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        self.fn = fn

    def __call__(self, input: Tensor) -> Tensor: return self.fn(input)


class HorizontalFlip(LambdaTransform):
    def __init__(self): super(HorizontalFlip, self).__init__(hflip)


class VerticalFlip(LambdaTransform):
    def __init__(self): super(VerticalFlip, self).__init__(vflip)


class Rotation90(LambdaTransform):
    def __init__(self): super(Rotation90, self).__init__(rot90deg)


class Rotation180(LambdaTransform):
    def __init__(self): super(Rotation180, self).__init__(rot180deg)


class Rotation270(LambdaTransform):
    def __init__(self): super(Rotation270, self).__init__(rot270deg)
