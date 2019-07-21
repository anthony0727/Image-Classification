import numpy as np


def load_cifar100():
    from keras.datasets.cifar100 import load_data
    (train_x, train_y), (test_x, test_y) = load_data()

    train_y = train_y.reshape((-1,))
    test_y = test_y.reshape((-1,))

    return (train_x, train_y), (test_x, test_y)


class Dataset(object):
    def __init__(self, images, labels):
        self.images = images.copy()
        self.labels = labels.copy().ravel()
        self.counter = 0

    def __len__(self):
        return len(self.images)

    def next_batch(self, batch_size=32):
        if self.counter + batch_size > len(self):
            self.shuffle()
            self.counter = 0

        i = self.counter
        batch_images = self.images[i:i + batch_size]
        batch_labels = self.labels[i:i + batch_size]

        self.counter += batch_size

        return batch_images.copy(), batch_labels.copy()

    def shuffle(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]


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
