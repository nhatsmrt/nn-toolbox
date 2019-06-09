from torchvision.transforms import functional as F

def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2)) / (c * h * w)


def is_image(filename):
    '''
    Check if filename has valid extension
    :param filename:
    :return: boolean indicating whether filename is a valid image filename
    '''
    filename = filename.lower()
    return filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")


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