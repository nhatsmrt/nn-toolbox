from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import Tensor
from .utils import is_image
from PIL import Image
import os
from typing import Tuple, Any, Optional


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
        self.transform = transform
        self._to_tensor = ToTensor()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, i) -> Tensor:
        if self.transform is not None:
            return self._to_tensor(self.transform(self._images[i]))
        else:
            return self._to_tensor(self._images[i])


class UnsupervisedFromSupervisedDataset(Dataset):
    '''
    Convert a supervisded dataset to an unsupervised dataset
    '''
    def __init__(self, dataset: Dataset, transform=None):
        self._data = dataset
        self.transform = transform

    def __getitem__(self, index):
        data = self._data.__getitem__(index)[0]
        return self.transform(data) if self.transform is not None else data

    def __len__(self):
        return len(self._data)


class PairedDataset(Dataset):
    '''
    Pair up two datasets, and allow users to sample a pair, one from each dataset
    '''
    def __init__(self, data_1: Dataset, data_2: Dataset):
        super(PairedDataset, self).__init__()
        self.data_1 = data_1
        self.data_2 = data_2

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        assert index < self.__len__()
        i = index % len(self.data_1)
        j = index // len(self.data_1)
        x1 = self.data_1[i]
        x2 = self.data_2[j]
        return x1, x2

    def __len__(self) -> int:
        return len(self.data_1) * len(self.data_2)

