import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from .utils import is_image
from PIL import Image
import os


class UnlabelledImageDataset(Dataset):
    def __init__(self, path, transform=None, device=None):
        '''
        :param path: path to folder of images
        :param transforms: A transform (possibly a composed one) taking in a PIL image and return a PIL image
        '''
        super(UnlabelledImageDataset, self).__init__()
        self._images = []
        for filename in os.listdir(path):
            if is_image(filename):
                full_path = path + filename
                self._images.append(Image.open(full_path))
        self._transform = transform
        self._to_tensor = ToTensor()
        self._device = device

    def __len__(self):
        return len(self._images)

    def __getitem__(self, i):
        if self._transform is not None:
            output = self._to_tensor(self._transform(self._images[i]))
        else:
            output = self._to_tensor(self._images[i])

        if self._device is not None:
            output.to(self._device)

        return output