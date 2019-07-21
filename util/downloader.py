from os import path


def download_imagenet():
    data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

    import urllib.request
    import tarfile

    urllib.request.urlretrieve(data_url, filename='./inception.tgz')

    tarfile.open(name='./inception.tgz', mode="r:gz").extractall('./inception_model')


if not path.isdir('./inception_model'):
    download_imagenet()
