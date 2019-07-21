# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from model.ABCNet import Network
from data import Cifar

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


def conv(layer, units, name):
    layer = tf.layers.Conv2D(units, (3, 3), (1, 1), 'SAME', activation=tf.nn.relu, name=name)(layer)

    return layer


def fc(layer):
    layer = tf.layers.Dense(1024, activation=tf.nn.relu)(layer)
    layer = tf.layers.Dropout(0.5)(layer, training=self.is_train)

    return layer


def vgg_block(filters, ith_block, layer, n_layers):
    for ith_layer in range(1, n_layers + 1):
        layer = conv(layer, filters, name='conv{}'.format(ith_layer))
    layer = tf.layers.MaxPooling2D((2, 2), (2, 2), name='MaxPool-{}'.format(ith_block))(layer)

    return layer


class VGGNet(Network):
    def __init__(self, n_layer=11):
        super(VGGNet, self).__init__()

        if n_layer not in vgg_config.keys():
            raise ValueError("Unrecognizable VGGNet. Only VGG11, VGG13, VGG16, VGG19 are available")

        self.n_layer = n_layer

    def attach_placeholders(self):
        self.xs = tf.placeholder(tf.float32, (None, *self.input_shape), name='xs')
        self.ys = tf.placeholder(tf.int32, (None,), name='ys')
        self.lr = tf.placeholder(tf.float32, (), name='lr')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

    def attach_layers(self):
        layer = self.xs

        config_zip = zip(vgg_config[self.n_layer], filters_config)
        for ith_block, (n_layers, filters) in enumerate(config_zip, start=1):
            with tf.variable_scope('VGGBLOCK-{}'.format(ith_block)):
                layer = vgg_block(filters, ith_block, layer, n_layers)

        with tf.variable_scope('FC'):
            layer = tf.layers.Flatten()(layer)
            layer = fc(layer)
            layer = fc(layer)
            logits = tf.layers.Dense(self.n_class)(layer)

        self.logits = tf.identity(logits, name='logits')
        self.y_pred = tf.nn.softmax(logits, name='y_pred')

        return layer

    def attach_loss(self):
        l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        l2_beta = 0.01

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys, logits=self.logits)
        loss = tf.reduce_mean(loss) + (l2_beta * l2_reg)

        self.loss = tf.identity(loss, 'loss')
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)  # must add to loss collection manually

    def attach_metric(self):
        with tf.variable_scope('metrics'):
            logits_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            self.accuracy = tf.metrics.accuracy(self.ys, logits_cls, name='accuracy')

            top5, top5_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(self.y_pred, self.ys, k=5), tf.float32) * 100)
            top1, top1_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(self.y_pred, self.ys, k=1), tf.float32) * 100)
            metric_loss, loss_op = tf.metrics.mean(self.loss)

            metric_init_op = tf.group(
                [var.initializer for var in self.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)],
                name='metric_init_op')
            metric_update_op = tf.group([top5_op, top1_op, loss_op], name='metric_update_op')

            top5 = tf.identity(top5, 'top5_accuracy')
            top1 = tf.identity(top1, 'top1_accuracy')
            tf.identity(metric_loss, 'metric_loss')

            self.attach_summary(top5, top1, metric_loss)

    def attach_summary(self, top5, top1, metric_loss):
        print(tf.GraphKeys.METRIC_VARIABLES)
        tf.summary.scalar('top5_tb', top5)
        tf.summary.scalar('top1_tb', top1)
        tf.summary.scalar('loss_tb', metric_loss)
        merged = tf.summary.merge_all(name='merge_all')
