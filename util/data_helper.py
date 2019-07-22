import numpy as np


def load_cifar100():
    from keras.datasets.cifar100 import load_data
    (train_x, train_y), (test_x, test_y) = load_data()

    train_y = train_y.reshape((-1,))
    test_y = test_y.reshape((-1,))

    return (train_x, train_y), (test_x, test_y)


def batch_generator(data, labels, batch_size=32):
    start_idx = 0
    num_step = len(data) // batch_size
    indexes = np.arange(0, len(data))
    while True:
        if start_idx >= num_step - 1:
            np.random.shuffle(indexes)
            start_idx = 0
        else:
            start_idx += 1
        batch_index = indexes[start_idx * batch_size:
                              (start_idx + 1) * batch_size]

        batch_data = data[batch_index]
        batch_label = labels[batch_index]

        yield batch_data, batch_label


class Dataset(object):
    def __init__(self, images, labels, n_batch=128):
        self.images = images.copy()
        self.labels = labels.copy().ravel()
        self.counter = 0
        self.generator = batch_generator(self.images, self.labels, batch_size=n_batch)

    def __len__(self):
        return len(self.images)

    def next_batch(self):
        return next(self.generator)

    def shuffle(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.images[indices] = self.images[indices]
        self.labels[indices] = self.labels[indices]


def random_crop_and_pad(images, pad=4):
    _, h, w, _ = images.shape
    pad_images = np.pad(images, [(0, 0), (pad, pad), (pad, pad), (0, 0)],
                        mode='constant')

    crops = []
    for idx, image in enumerate(pad_images):
        start_y = np.random.randint(0, pad)
        start_x = np.random.randint(0, pad)

        cropped = image[start_y:start_y + h, start_x:start_x + w]

        crops.append(cropped)

    return np.stack(crops)


def random_flip_left_right(images):
    for idx, image in enumerate(images):
        if np.random.random() > 0.5:
            images[idx] = images[idx, :, ::-1]

    return images


def augment_images(images):
    images = random_crop_and_pad(images, pad=4)
    images = random_flip_left_right(images)

    return images
