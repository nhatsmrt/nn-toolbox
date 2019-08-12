from torchvision.transforms import functional as F
from torch import Tensor
import cv2
from cv2 import imread, cvtColor
from numpy import ndarray


__all__ = ['gram_matrix', 'is_image', 'pil_to_tensor', 'tensor_to_pil', 'tensor_to_np', 'cv2_read_image']


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2)) / (c * h * w)


def is_image(filename):
    """
    Check if filename has valid extension

    :param filename:
    :return: boolean indicating whether filename is a valid image filename
    """
    filename = filename.lower()
    return filename.endswith(".jpg") \
           or filename.endswith(".png") \
           or filename.endswith(".jpeg") \
           or filename.endswith(".gif") \
           or filename.endswith(".bmp")


def pil_to_tensor(pil, device=None):
    tensor = F.to_tensor(pil).unsqueeze(0)
    if device is not None:
        tensor.to(device)

    return tensor


def tensor_to_pil(tensor):
    if len(tensor.shape) == 4:
        return F.to_pil_image(tensor[0])
    else:
        return F.to_pil_image(tensor)


def tensor_to_np(tensor: Tensor) -> ndarray:
    """Convert the tensor image to numpy format"""
    if len(tensor.shape) == 4:
        return tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
    else:
        return tensor.permute(1, 2, 0).cpu().detach().numpy()


def cv2_read_image(image_path, to_float: bool=False, flag: int=cv2.IMREAD_COLOR) -> ndarray:
    """
    Read an image using cv2 and convert to RGB

    :param image_path:
    :param to_float: whether to convert image to float dats type:
    :param flag: indicate mode for cv2 read image
    :return:
    """
    assert is_image(image_path)
    img = imread(image_path, flag)
    img = cvtColor(img, cv2.COLOR_BGR2RGB)
    if to_float:
        img = img / 255
    return img


# def is_color(image, batch: bool=True) -> bool:
#     """
#     Check if image(s) is colored properly (i.e has 4 channels)
#     :param image:
#     :param batch:
#     """
#     if batch:
#         return len(image.shape) == 4 and
#     return len(image.shape) == 3 if
