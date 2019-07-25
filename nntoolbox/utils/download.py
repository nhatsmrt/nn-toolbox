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
    print(url)
    req = requests.get(url, stream=True)
    print(req)
    if req.status_code == 404:
        raise ConnectionError("Request invalid")
    elif int(req.headers.get('content-Length')) > max_size:
        raise ConnectionError("File too large")
    else:
        with open(filename, 'wb') as f:
            # for chunk in req:
            #     f.write(chunk)
            f.write(req.content)

