import numpy as np
import numpy.random as npr

from abc import abstractmethod


class Image:
    def __init__(self):
        pass

    def augment(self, num_data):
        # how much to augment?
        pass

    @staticmethod
    def random_crop(images, pad=4):
        _, h, w, _ = images.shape

        pad_images = np.pad(images, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')

        crops = []
        for i, img in enumerate(pad_images):
            x, y = npr.randint(0, 2 * pad - 1), npr.randint(0, 2 * pad - 1)
            crop = img[y:y + h, x:x + w]
            crops.append(crop)

        return np.stack(crops)

    @staticmethod
    def random_flip_up_down(images):

        size = npr.randint(len(images) - 1)
        indices = npr.choice(range(len(images)), size)

        images[indices] = images[indices, ::-1, :]

        return images

    @staticmethod
    def random_flip_left_right(images):
        size = npr.randint(len(images) - 1)
        indices = npr.choice(range(len(images)), size)

        images[indices] = images[indices, :, ::-1]

        return images

        pass

    def random_rotate(self, images, max_angle=30):
        _, h, w, _ = images.shape
        for i, img in enumerate(images):
            angle = npr.randint(-max_angle, max_angle)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            images[i] = cv2.warpAffine(img, M, (w, h))

        return images

    def random_hue(self):
        pass

    def random_saturation(self):
        pass

    def random_brightness(self):
        pass


class Cifar10(Image):
    """ wrapper class for keras.dataset.cifar10

    usage:
        data = Cifar10()
        gen = Cifar10.generator(data.x_train, data.y_train)
        next(gen)

    """

    def __init__(self):
        super(Cifar10, self).__init__()

        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # minmax normalize
        self.x_train, self.x_test = x_train / 255, x_test / 255
        self.y_train, self.y_test = y_train.squeeze(), y_test.squeeze()

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

    @classmethod
    def generator(cls, data, labels, batch_size=32):
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
