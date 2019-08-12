from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image
import numpy as np
import collections


__all__ = ['ElasticDeformation', 'Cutout']


class ElasticDeformation(object):
    """
    Apply elastic deformation on a PIL image (H x W x C)
    Adapt from https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
    Paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.5032&rep=rep1&type=pdf
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image: Image)->Image:
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return ElasticDeformation.elastic_deform(image, alpha=alpha, sigma=sigma)

    @staticmethod
    def elastic_deform(image:Image, alpha=1000, sigma=30, spline_order=1, mode='nearest') -> Image:
        """Elastic deformation of image as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
           :param image: The image to be deformed
           :param alpha:  scaling factor that controls the intensity of the deformation
           :param sigma: the std of gaussian filters. Smaller sigma implies more random deformation field
           :param spline_order
           :param mode: interpolation mode
        """

        image = np.array(image)
        shape = image.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        result = np.empty_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = map_coordinates(
                image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
        return Image.fromarray(result)


class Cutout(object):
    '''
    https://arxiv.org/pdf/1708.04552.pdf
    '''
    def __init__(self, n_holes, length):
        self._n_holes = n_holes
        self._length = length

    def __call__(self, image: Image) -> Image:
        h, w = image.size
        ret = np.array(image)

        for _ in range(self._n_holes):
            h1 = np.random.choice(h)
            h2 = min(h, h1 + self._length)

            w1 = np.random.choice(w)
            w2 = min(w, w1 + self._length)

            ret[h1:h2, w1:w2] = 0

        return Image.fromarray(ret)


def random_num_generator(config):
    if config[0] == 'uniform':
        ret = np.random.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = np.random.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret
