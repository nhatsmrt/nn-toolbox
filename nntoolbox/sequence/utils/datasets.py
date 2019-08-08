from torchtext.datasets import TranslationDataset
from torchtext.data import Field
from typing import Tuple
from nntoolbox.utils import download_from_url
import tarfile
import os


__all__ = ['Europarl']


class Europarl(TranslationDataset):
    """
    European Parliament Proceedings Parallel Corpus 1996-2011 (UNTESTED)
    """
    base_url = "http://www.statmt.org/europarl/v7/"

    def __init__(self, root: str, exts: Tuple[str, str], fields: Tuple[Field, Field], download: bool=False):
        """
        :param root: directory containing the europarl-v7 file
        :param exts: extension denoting languages
        :param fields: 2 text fields, one for source and one for target
        :param download: whether to download the data
        """
        assert isinstance(exts, tuple) and len(exts) == 2 and '.en' in exts and exts[0] != exts[1]
        assert isinstance(fields, tuple) and len(fields) == 2

        non_en = exts[0][1:] if exts[1] == ".en" else exts[1][1:]

        path = root + "europarl-v7." + non_en + "-" + "en"

        if download and not (os.path.exists(path + "." + non_en) and os.path.exists(path + ".en")):
            print("Downloading data.")
            url = self.base_url + non_en + "-" + "en.tgz"
            tar_path = root + non_en + "-" + "en.tgz"
            download_from_url(url, tar_path, max_size=None)
            with tarfile.open(tar_path) as tar:
                tar.extractall(root)
        else:
            print("Data already downloaded.")

        super(Europarl, self).__init__(path, exts, fields)
