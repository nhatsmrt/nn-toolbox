import requests
from typing import Optional


__all__ = ['download_from_url']


def download_from_url(url: str, filename: str, max_size: Optional[int]=50):
    """
    Download from a url, and save to filename

    :param url:
    :param filename:
    :param max_size: (in kbs)
    :return:
    """
    if max_size is not None: max_size *= 1024
    req = requests.get(url, stream=max_size is not None)
    if req.status_code == 404:
        raise ConnectionError("Request invalid")
    elif max_size is not None and int(req.headers.get('content-Length')) > max_size:
        raise ConnectionError("File too large")
    else:
        with open(filename, 'wb') as f:
            # for chunk in req:
            #     f.write(chunk)
            f.write(req.content)

