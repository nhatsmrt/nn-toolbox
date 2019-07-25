import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple
from torch import Tensor
from torchvision.transforms import ToTensor
from PIL import Image
from warnings import warn
from ...utils import download_from_url


__all__ = ['FaceScrub']


class FaceScrub(Dataset):
    def __init__(self, root, data_path, transform=None):
        self.images_paths = []
        self.labels = []
        self.name2idx = dict()
        self.idx2name = dict()
        self.transform = ToTensor() if transform is None else transform

        df_male = pd.read_csv(root + "/facescrub_actors.txt", sep='\t')
        for i in range(len(df_male)):
            try:
                url = df_male['url'][i]
                print(url)
                path = data_path + "/face_" + str(len(self.images_paths)) + url[:-4]
                download_from_url(url, path)
                # Image.open(path)
            except:
                warn("Image corrupted or URL error. Skip to next image.")
            else:
                self.images_paths.append(path)
                name = df_male['name'][i]
                if name not in self.idx2name:
                    idx = len(self.name2idx)
                    self.name2idx[name] = idx
                    self.idx2name[idx] = name

                self.labels.append([self.name2idx[name]])

                if len(self.images_paths) >= 100:
                    break

        df_female = pd.read_csv(root + "/facescrub_actresses.txt", sep='\t')
        for i in range(len(df_female)):
            try:
                url = df_female['url'][i]
                path = data_path + "/face_" + str(len(self.images_paths)) + url[:-4]
                download_from_url(url, path)
                # Image.open(path)
            except:
                warn("Image corrupted or URL error. Skip to next image.")
            else:
                self.images_paths.append(path)
                name = df_female['name'][i]
                if name not in self.idx2name:
                    idx = len(self.name2idx)
                    self.name2idx[name] = idx
                    self.idx2name[idx] = name

                self.labels.append([self.name2idx[name]])
                if len(self.images_paths) >= 100:
                    break

    def __len__(self) -> int: return len(self.images_paths)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        image = Image.open(self.images_paths[i])
        image = image.convert('RGB')
        return self.transform(image), self.labels[i]
