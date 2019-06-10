import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from .utils import is_image
from PIL import Image
import os


class UnlabelledImageDataset(Dataset):
    def __init__(self, path, transform=None, img_dim=None):
        '''
        :param path: path to folder of images
        :param transforms: A transform (possibly a composed one) taking in a PIL image and return a PIL image
        '''
        print("Begin reading images and convert to RGB")
        super(UnlabelledImageDataset, self).__init__()
        self._images = []
        for filename in os.listdir(path):
            if is_image(filename):
                full_path = path + filename
                image = Image.open(full_path).convert('RGB')
                if img_dim is not None:
                    image = image.resize(img_dim)
                self._images.append(image)
        self._transform = transform
        self._to_tensor = ToTensor()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, i):
        if self._transform is not None:
            return self._to_tensor(self._transform(self._images[i]))
        else:
            return self._to_tensor(self._images[i])