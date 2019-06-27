# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from ABCNet import Network
from data import Cifar10

N_VGGBLOCK = 5

vgg11_config = [1, 1, 2, 2, 2]
vgg13_config = [2, 2, 2, 2, 2]
vgg16_config = [2, 2, 3, 3, 3]
vgg19_config = [2, 2, 4, 4, 4]

vgg_config = {
    11: vgg11_config,
    13: vgg13_config,
    16: vgg16_config,
    19: vgg19_config
}

filters_config = [64, 128, 256, 512, 512]


def conv(prev_layer, units, name):
    layer = tf.layers.Conv2D(units, (3, 3), (1, 1), 'SAME', activation=tf.nn.relu, name=name)(prev_layer)
    return layer


def fc(flat_layer, units, activation, initializer, layer_name):
    layer = tf.layers.Dense(units, activation, kernel_initializer=initializer, name=layer_name)(flat_layer)
    return layer


class VGGNet(Network):
    def __init__(self, n_layer=11):
        super(VGGNet, self).__init__()

        self.input_shape = None
        self.n_class = 0

        if n_layer not in vgg_config.keys():
            raise ValueError("Unrecognizable VGGNet. Only VGG11, VGG13, VGG16, VGG19 are available")

        self.n_layer = n_layer

    def transfer(self, network):
        pass

    def attach_placeholders(self):
        self.xs = tf.placeholder(tf.float32, (None, *self.input_shape), name='xs')
        self.ys = tf.placeholder(tf.float32, (None,), name='ys')
        self.lr = tf.placeholder(tf.float32, (), name='lr')
        self.is_train = tf.placeholder(tf.bool, name='phase_train')

    def attach_layers(self):
        layer = self.xs

        config_zip = zip(vgg_config[self.n_layer], filters_config)
        for ith_block, (n_layers, filters) in enumerate(config_zip, start=1):
            with tf.variable_scope('VGGBLOCK-{}'.format(ith_block)):
                for ith_layer in range(1, n_layers + 1):
                    layer = conv(layer, filters, name='conv{}'.format(ith_layer))
                layer = tf.layers.MaxPooling2D((2, 2), (2, 2), name='MaxPool-{}'.format(ith_block))(layer)

        with tf.variable_scope('FC'):
            layer = tf.layers.Flatten()(layer)
            layer = tf.layers.Dense(1024, activation=tf.nn.relu)(layer)
            layer = tf.layers.Dropout(0.5)(layer, training=self.is_train)
            logits = tf.layers.Dense(self.n_class)(layer)

        self.logits = tf.identity(logits, name='logits')
        pred = tf.nn.softmax(logits, name='predictions')

    def attach_loss(self):
        l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        l2_beta = 0.01

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.logits)
        loss = tf.reduce_mean(loss) + (l2_beta * l2_reg)

        self.loss = tf.identity(loss, 'loss')
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    def attach_metric(self):
        logits_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
        self.accuracy, _ = tf.metrics.accuracy(self.ys, logits_cls, name='accuracy')

    def attach_summary(self):
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.loss)
        tf.summary.merge_all(name='merge_all')


net = VGGNet(13)
net.build((32, 32, 3), 10)

for var in net.graph.get_collection('variables'):
    print(var)
