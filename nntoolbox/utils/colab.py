__all__ = ['download', 'download_folder']


def download(drive, id, name):
    downloaded = drive.CreateFile({'id': id})
    downloaded.GetContentFile(name)


def download_folder(drive, id):
    file_list = drive.ListFile(
        {'q': "'" + id + "' in parents"}).GetList()  # use your own folder ID here

    for f in file_list:
        # 3. Create & download by id.
        print('title: %s, id: %s' % (f['title'], f['id']))
        fname = f['title']
        print('downloading to {}'.format(fname))
        f_ = drive.CreateFile({'id': f['id']})
        f_.GetContentFile(fname)
