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


def vggblock():
    pass


def fc(flat_layer, units, activation, initializer, layer_name):
    layer = tf.layers.Dense(units, activation, kernel_initializer=initializer, name=layer_name)(flat_layer)
    return layer


class VGGNet(Network):
    def train(self, optimizer):
        pass

    def __init__(self):
        super(VGGNet, self).__init__()

    def __attach_vggblock(self):
        pass

    def attach_placeholders(self, shape):
        with self.graph.as_default():
            # define input placeholder
            tf.placeholder(tf.float32, (None, *shape), name='xs')  # fix me #
            tf.placeholder(tf.float32, (None,), name='ys')  # fix me #
            tf.placeholder(tf.float32, (), name='lr')  # fix me #
            tf.placeholder(tf.bool, name='phase_train')  # fix me #

    def attach_layers(self):
        with self.graph.as_default():
            xs = self.graph.get_tensor_by_name('xs:0')
            phase_train = self.graph.get_tensor_by_name('phase_train:0')

            #########################################################

            layer1 = self.conv(xs, 64, (3, 3), (1, 1), 'SAME', tf.nn.relu, 'layer1')

            layer2 = self.conv(layer1, 128, (3, 3), (2, 2), 'SAME', tf.nn.relu, 'layer2')

            top_conv = tf.identity(layer2, 'top_conv')

            flat_layer = tf.layers.Flatten()(top_conv)

            fc_initializer = tf.initializers.glorot_normal  # fix me#
            fc_layer_1 = fc(flat_layer, 256, tf.nn.relu, fc_initializer, 'fc_layer_1')
            fc_layer_1 = tf.layers.dropout(fc_layer_1, training=phase_train, rate=0.7)

            fc_layer_2 = fc(fc_layer_1, 256, tf.nn.relu, fc_initializer, 'fc_layer_2')
            fc_layer_2 = tf.layers.dropout(fc_layer_2, training=phase_train, rate=0.7)

            outputs = fc(fc_layer_2, 10, None, None, 'outputs')
            tf.identity(outputs, 'logits')

    def attach_loss(self):
        with self.graph.as_default():
            labels = self.graph.get_tensor_by_name('ys:0')
            logits = self.graph.get_tensor_by_name('logits:0')

            l2_reg = tf.reduce_sum(
                [tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])  # fix me #
            l2_beta = 0.01

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
            loss = tf.reduce_mean(loss, name='loss') + (l2_beta * l2_reg)
            tf.identity(loss, 'loss')

            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    def attach_metric(self):
        # metric
        with self.graph.as_default():
            labels = self.graph.get_tensor_by_name('ys:0')
            logits = self.graph.get_tensor_by_name('logits:0')

            logits_cls = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            tf.metrics.accuracy(labels, logits_cls, name='accuracy')  # you need to init local vars for this

    def attach_summary(self):
        with self.graph.as_default():
            acc = self.graph.get_tensor_by_name('accuracy:0')
            loss = self.graph.get_tensor_by_name('loss:0')

            tf.summary.scalar('accuracy', acc)
            tf.summary.scalar('loss', loss)
            tf.summary.merge_all(name='merge_all')

    def build(self, shape):
        self.attach_placeholders(shape)
        self.attach_layers()
        self.attach_loss()
        self.attach_metric()
        self.attach_summary()

    def fit(self, data):
        self.build(data.x_shape)

        from util import train
        self.graph = train(self, data)

    def transfer(self, network):
        pass
