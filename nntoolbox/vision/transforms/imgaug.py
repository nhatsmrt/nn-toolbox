"""
Wrapper for imgaug transformation
"""
from torch import tensor, Tensor
import numpy as np
from numpy import ndarray
from imgaug.augmenters import Augmenter


__all__ = ['ImgAugTransform', 'NumpyToTensor', 'ToFloat']


class ImgAugTransform:
    """A thin wrapper around img aug transform"""
    def __init__(self, transform: Augmenter):
        self.transform = transform

    def __call__(self, image: ndarray) -> ndarray:
        """
        :param image: must be an uint8 image or a batch of uint8 image
        :return:
        """
        assert isinstance(image, ndarray)
        assert image.dtype == np.uint8
        if len(image.shape) == 4:
            return np.concatenate(list(self.transform.augment_batches([[im] for im in image])), axis=0)
        else:
            return self.transform.augment_image(image)


class NumpyToTensor:
    """
    A simple class to transform a (batch) of numpy images to tensor
    """

    def __call__(self, image: ndarray) -> Tensor:
        img_tensor = tensor(image)
        if len(img_tensor.shape) == 4:
            img_tensor = img_tensor.permute(0, 3, 1, 2)
        elif len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor


class ToFloat:
    """
    A simple transform to convert image(s) from uint8 to float
    """

    def __call__(self, image):
        image = image.float() if isinstance(image, Tensor) else image.astype(np.float32)
        return image / 255
