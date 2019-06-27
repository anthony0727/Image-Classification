# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from ABCNet import Network
from data import Cifar10


def conv(input_xs, units, k, s, padding, activation, name):
    layer = tf.layers.Conv2D(units, k, s, padding, activation=activation, name=name)(input_xs)
    return layer


def fc(flat_layer, units, activation, initializer, layer_name):
    layer = tf.layers.Dense(units, activation, kernel_initializer=initializer, name=layer_name)(flat_layer)
    return layer


def vgg_block(i, filters, layer):
    with tf.variable_scope('VGGBLOCK-{}'.format(i)):
        layer = tf.layers.Conv2D(filters, (3, 3), (1, 1), 'SAME', activation=tf.nn.relu, name='conv1')(layer)
        layer = tf.layers.Conv2D(filters, (3, 3), (1, 1), 'SAME', activation=tf.nn.relu, name='conv2')(layer)
    layer = tf.layers.MaxPooling2D((2, 2), (2, 2), name='MaxPool-{}'.format(i))(layer)

    return layer


class VGGNet(Network):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.input_shape = None
        self.num_class = 0

    def build(self, input_shape, num_class):
        self.input_shape = input_shape
        self.num_class = num_class

        self.attach_placeholders()
        self.attach_layers()
        self.attach_loss()
        self.attach_metric()
        self.attach_summary()

    def transfer(self, network):
        pass

    def train(self, optimizer):
        pass

    def attach_placeholders(self):
        with self.graph.as_default():
            # define input placeholder
            self.xs = tf.placeholder(tf.float32, (None, *self.shape), name='xs')
            self.ys = tf.placeholder(tf.float32, (None,), name='ys')
            self.lr = tf.placeholder(tf.float32, (), name='lr')
            self.is_train = tf.placeholder(tf.bool, name='phase_train')

    def attach_layers(self):
        with self.graph.as_default():
            with tf.variable_scope('VGGBLOCK-1'):
                layer = tf.layers.Conv2D(32, (3, 3), (1, 1), 'SAME', activation=tf.nn.relu, name='conv1')(self.xs)
            with tf.variable_scope('VGGBLOCK-2'):
                layer = tf.layers.Conv2D(64, (3, 3), (1, 1), 'SAME', activation=tf.nn.relu, name='conv1')(layer)

            # 중복 코드 도저히 못참겠다...
            filter_config = [128, 256, 256]
            for i in range(3):
                block_num = i+3
                layer = vgg_block(block_num, filter_config[i], layer)

            with tf.variable_scope('FC'):
                layer = tf.layers.Flatten()(layer)
                layer = tf.layers.Dense(1024, activation=tf.nn.relu)(layer)
                layer = tf.layers.Dropout(0.5)(layer, training=self.is_train)
                logits = tf.layers.Dense(self.num_class)(layer)

            self.logits = tf.identity(logits, name='logits')
            pred = tf.nn.softmax(logits, name='predictions')

    def attach_loss(self):
        with self.graph.as_default():
            l2_reg = tf.reduce_sum(
                [tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])  # fix me #
            l2_beta = 0.01

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.ys, self.logits)
            loss = tf.reduce_mean(loss) + (l2_beta * l2_reg)

            self.loss = loss

            tf.identity(loss, 'loss')
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    def attach_metric(self):
        with self.graph.as_default():
            logits_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            tf.metrics.accuracy(self.ys, logits_cls, name='accuracy')  # you need to init local vars for this

    def attach_summary(self):
        with self.graph.as_default():
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)
            tf.summary.merge_all(name='merge_all')
