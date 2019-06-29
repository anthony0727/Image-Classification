import numpy as np
import numpy.random as npr

import tensorflow as tf


def image_augmentation(image, is_training, crop_h, crop_w):
    def _aug_with_train(input_x, crop_height, crop_width):
        img_h, img_w, ch = list(map(int, input_x.get_shape()[:]))

        pad_w = int(img_h * 0.4)
        pad_h = int(img_w * 0.4)

        input_x = tf.image.resize_image_with_crop_or_pad(input_x, img_h + pad_h, img_w + pad_w)
        input_x = tf.image.random_crop(input_x, [crop_height, crop_width, ch])
        input_x = tf.image.random_flip_left_right(input_x)
        input_x = tf.image.random_flip_up_down(input_x)

        input_x = tf.image.random_hue(input_x, max_delta=63. / 255.)
        input_x = tf.image.random_brightness(input_x, max_delta=63. / 255.)
        input_x = tf.image.random_saturation(input_x, lower=0.5, upper=1.8)

        input_x = tf.image.per_image_standardization(input_x)

        return input_x

    def _aug_with_test(input_x, crop_height, crop_width):
        input_x = tf.image.resize_image_with_crop_or_pad(input_x, crop_height, crop_width)
        input_x = tf.image.per_image_standardization(input_x)

        return input_x

    image = tf.cond(is_training,
                    lambda: _aug_with_train(image, crop_h, crop_w),
                    lambda: _aug_with_test(image, crop_h, crop_w))
    return image


def images_augmentation(images, is_train):
    crop_h, crop_w = list(map(int, images.shape[1:3]))
    images = tf.map_fn(lambda image: image_augmentation(image, is_train, crop_h, crop_w), images)
    return images


def generator(data, labels, batch_size=32):
    """
    usage:
        gen = generator(x_train, y_train)
        next(gen)
    """

    idx = 0
    num_step = len(data) // batch_size
    indexes = np.arange(0, len(data))
    while True:
        if idx >= num_step - 1:
            npr.shuffle(indexes)
            idx = 0
        else:
            idx += 1
        batch_index = indexes[idx * batch_size:
                              (idx + 1) * batch_size]

        batch_data = data[batch_index]
        batch_label = labels[batch_index]

        yield batch_data, batch_label


class Cifar:
    """ wrapper class for keras.dataset.cifar10
    """

    def __init__(self, n_class):
        if n_class == 10:
            from keras.datasets import cifar10
            cifar = cifar10
        elif n_class == 100:
            from keras.datasets import cifar100
            cifar = cifar100

        self.n_class = n_class

        (x_train, y_train), (x_test, y_test) = cifar.load_data()


        # minmax normalize
        self.x_train, self.x_test = (x_train / 255.).astype(np.float32), (x_test / 255.).astype(np.float32)
        self.y_train, self.y_test = y_train.squeeze(), y_test.squeeze()
        print(self.x_train.dtype, self.x_test.dtype)

        # keras.dataset guarantees data shape consistency
        self.x_shape = self.x_train.shape[1:]
        self.y_shape = self.y_train.shape[1:]

        print('image shape : {}, label shape : {} '.format(x_train.shape, y_train.shape))
        print('image shape : {}, label shape : {} '.format(x_test.shape, y_test.shape))
        print('train minimun : {}, train_maximum : {} '.format(x_train.min(), x_train.max()))
        print('tests minimun : {}, test_maximum : {} '.format(x_test.min(), x_test.max()))

    def __getattr__(self, item):
        if item == 'x_shape':
            return self.x_shape
        elif item == 'y_shape':
            return self.y_shape
