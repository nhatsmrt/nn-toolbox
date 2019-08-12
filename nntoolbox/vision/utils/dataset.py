import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple
from torch import Tensor
from torchvision.transforms import ToTensor
from PIL import Image
from warnings import warn
from ...utils import download_from_url
import os


__all__ = ['FaceScrub', 'download_facescrub']


def download_facescrub(root: str, data_path: str, max_size: int=128):
    print("bleh")
    df_male = pd.read_csv(root + "/facescrub_actors.txt", sep='\t')
    df_female = pd.read_csv(root + "/facescrub_actresses.txt", sep='\t')

    n_image = 0
    n_ppl = 0

    df_both = pd.concat([df_male, df_female])

    for i in range(len(df_both)):
        try:
            url = df_both['url'][i]
            name = df_both['name'][i]

            folder = data_path + "/" + name
            if not os.path.exists(folder):
                os.makedirs(folder)
                n_ppl += 1

            path = folder + "/face_" + str(n_image) + url[:-4]
            download_from_url(url, path, max_size)
            # Image.open(path)
        except:
            warn("Image corrupted or URL error. Skip to next image.")
        else:
            n_image += 1

            if n_image >= 2: break

    print("Finish downloading " + str(n_image) + " images of " + str(n_ppl) + " people.")


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
