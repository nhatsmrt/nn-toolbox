from typing import Union, Tuple
from torchvision.transforms.functional import resize
import numpy as np
from PIL import Image


__all__ = ['RandomRescale']


class RandomRescale:
    """Randomly downscale an image, then resize it back to original scale"""
    def __init__(self, scale: Union[float, Tuple[float, float]]=(0.5, 1.0)):
        self.scale = scale if isinstance(scale, tuple) else (scale, scale)

    def __call__(self, image: Image) -> Image:
        width, height = image.size
        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        new_width, new_height = int(width * scale), int(height * scale)
        return resize(resize(image, (new_height, new_width)), (height, width))
