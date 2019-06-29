from abc import ABC, abstractmethod
import tensorflow as tf


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
            self.is_train = tf.placeholder_with_default(True, None)
            self.saver = None

            # tuple
            self.input_shape = None

            # integer
            self.n_class = 0
            self.train_writer = tf.summary.FileWriter('./train_tb')
            self.test_writer = tf.summary.FileWriter('./test_tb')

    def build(self, input_shape, num_class):
        self.input_shape = input_shape
        self.n_class = num_class

        with self.graph.as_default():
            self.attach_placeholders()
            self.attach_layers()
            self.attach_loss()
            self.attach_metric()
            self.attach_summary()
            self.saver = tf.train.Saver()

    def transfer(self):
        pass

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

    @abstractmethod
    def transfer(self):
        pass
