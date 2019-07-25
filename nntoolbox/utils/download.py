import requests


__all__ = ['download_from_url']


def download_from_url(url: str, filename: str, max_size: int=50):
    """
    Download from a url, and save to filename

    :param url:
    :param filename:
    :param max_size: (in kbs)
    :return:
    """
    max_size *= 1024
    req = requests.get(url)
    if req.status_code == 404:
        raise ConnectionError("Request invalid")
    elif int(req.headers.get('Content-Length')) > max_size:
        raise ConnectionError("File too large")
    else:
        with open(filename, 'wb') as f:
            f.write(req.content)
