"""Transform input by batch"""
from torch import Tensor, flip, rot90
from functools import partial
from typing import List, Callable


__all__ = [
    'BatchCompose', 'Identity', 'BatchLambdaTransform',
    'BatchHorizontalFlip', 'BatchVerticalFlip', 'BatchRotation90',
    'BatchRotation180', 'BatchRotation270'
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


class BatchLambdaTransform:
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        self.fn = fn

    def __call__(self, input: Tensor) -> Tensor: return self.fn(input)


class BatchHorizontalFlip(BatchLambdaTransform):
    def __init__(self): super(BatchHorizontalFlip, self).__init__(hflip)


class BatchVerticalFlip(BatchLambdaTransform):
    def __init__(self): super(BatchVerticalFlip, self).__init__(vflip)


class BatchRotation90(BatchLambdaTransform):
    def __init__(self): super(BatchRotation90, self).__init__(rot90deg)


class BatchRotation180(BatchLambdaTransform):
    def __init__(self): super(BatchRotation180, self).__init__(rot180deg)


class BatchRotation270(BatchLambdaTransform):
    def __init__(self): super(BatchRotation270, self).__init__(rot270deg)
