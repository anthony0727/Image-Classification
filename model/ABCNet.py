from abc import ABC, abstractmethod
import tensorflow as tf


class Network(ABC):
    def __init__(self):
        self.xs, self.ys, self.lr, \
        self.loss, self.accuracy, \
        self.logits, self.is_train = \
            None, None, None, None, None, None, None

        self.graph = tf.Graph()
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter('./train_tb')
        self.test_writer = tf.summary.FileWriter('./test_tb')

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
    def build(self):
        pass

    @abstractmethod
    def train(self, optimizer):
        pass

    @abstractmethod
    def transfer(self):
        pass
