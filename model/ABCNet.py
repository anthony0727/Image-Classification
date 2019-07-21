from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class Network(ABC):
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            # tensors
            self.xs = None
            self.ys = None
            self.lr = None
            self.loss = None
            self.accuracy = None
            self.logits = None
            self.y_pred = None
            self.is_train = tf.placeholder_with_default(True, None)

            # tuple
            self.input_shape = None

            # integer
            self.n_class = 0

    def build(self, input_shape, num_class):
        self.input_shape = input_shape
        self.n_class = num_class

        with self.graph.as_default():
            self.attach_placeholders()
            self.attach_layers()
            self.attach_loss()
            self.attach_metric()

    def transfer_ops(self, weights):
        ops = []

        with self.graph.as_default():
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for i, layer in enumerate(trainable_variables):
                w_shape = weights[i].shape
                op = tf.assign(layer[:w_shape[0], :w_shape[1], :w_shape[2], :w_shape[3]],
                               weights[i])
                ops.append(op)

        return ops

    @abstractmethod
    def attach_placeholders(self):
        pass

    @abstractmethod
    def attach_layers(self):
        pass

    @abstractmethod
    def attach_loss(self):
        pass

    @abstractmethod
    def attach_metric(self):
        pass

    @abstractmethod
    def attach_summary(self):
        pass
