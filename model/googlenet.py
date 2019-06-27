from abc import ABC

import tensorflow as tf
import pandas as pd
import numpy as np
import numpy.random as npr

import cv2
import matplotlib.pyplot as plt

from ABCNet import Network

input_shape = (224, 224, 3)
num_classes = 1000

graph = tf.Graph()

with graph.as_default():
    images = tf.placeholder(tf.float32, (None, *input_shape), name='images')
    labels = tf.placeholder(tf.int32, (None,), name='labels')

with tf.variable_scope('preprocess'):
    image_mean = tf.constant([123.68, 116.779, 103.939])
    x = images - image_mean

he_init = tf.initializers.he_uniform()
xavier_init = tf.initializers.glorot_normal()


def inception_module(prev_layer,
                     one_conv_filters,
                     reduced_three_filters,
                     three_conv_filters,
                     reduced_five_filters,
                     five_conv_filters,
                     after_pool_filters,
                     module_name='inception'):
    with tf.variable_scope(module_name):
        out1 = tf.layers.Conv2D(one_conv_filters, (1, 1), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='1x1_conv')(prev_layer)

        out2 = tf.layers.Conv2D(reduced_three_filters, (1, 1), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='3x3_reduced')(prev_layer)
        out2 = tf.layers.Conv2D(three_conv_filters, (3, 3), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='3x3_conv')(out2)

        out3 = tf.layers.Conv2D(reduced_five_filters, (1, 1), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='5x5_reduced')(prev_layer)
        out3 = tf.layers.Conv2D(five_conv_filters, (5, 5), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='5x5_conv')(out3)

        out4 = tf.layers.MaxPooling2D((3, 3), (1, 1), padding='SAME', kernel_initializer=he_init,
                                      activation=tf.nn.relu, name='3x3_pool')(prev_layer)
        out4 = tf.layers.Conv2D(after_pool_filters, (5, 5), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='after_pool')(out4)

        out = tf.concat([out1, out2, out3, out4], axis=-1, name='filter_concatenation')

    return out


def auxiliary_network(block_4a, block_4d):
    with tf.variable_scope('auxiliary_network_4a'):
        avg_pool = tf.layers.AveragePooling2D((5, 5), (3, 3))(block_4a)
        conv = tf.layers.Conv2D(128, (1, 1), kernel_initializer=he_init, activation=tf.nn.relu, name='1x1')(
            avg_pool)

        fc = tf.layers.Flatten()(conv)
        fc = tf.layers.Dense(1024, kernel_initializer=he_init, activation=tf.nn.relu)(fc)
        fc = tf.layers.Dropout(0.7)(fc)
        aux_logit_4a = tf.layers.Dense(1000, kernel_initializer=xavier_init)(fc)

    with tf.variable_scope('auxiliary_network_4d'):
        avg_pool = tf.layers.AveragePooling2D((5, 5), (3, 3))(block_4d)
        conv = tf.layers.Conv2D(128, (1, 1), kernel_initializer=he_init, activation=tf.nn.relu, name='1x1')(
            avg_pool)

        fc = tf.layers.Flatten()(conv)
        fc = tf.layers.Dense(1024, kernel_initializer=he_init, activation=tf.nn.relu)(fc)
        fc = tf.layers.Dropout(0.7)(fc)
        aux_logit_4d = tf.layers.Dense(1000, kernel_initializer=xavier_init)(fc)

    return aux_logit_4a, aux_logit_4d


class GoogLeNet(Network, ABC):
    def __init__(self):
        super(GoogLeNet, self).__init__()

    def build(self):
        pass

    def train(self, optimizer):
        pass

    def transfer(self):
        pass

    def attach_layers(self):
        with self.graph.as_default():
            he_init = tf.initializers.he_uniform()

            conv1 = tf.layers.Conv2D(64, (7, 7), (2, 2), padding='SAME',
                                     kernel_initializer=he_init,
                                     name='7x7_conv')(x)
            pool1 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MaxPool_1')(conv1)
            conv2 = tf.layers.Conv2D(192, (3, 3), padding='SAME',
                                     kernel_initializer=he_init,
                                     name='3x3_conv')(pool1)

            pool2 = tf.layers.MaxPooling2D((3, 3), (2, 2),
                                           name='MaxPool_2')(conv2)

        with graph.as_default():
            block_3a = inception_module(pool2, 64, 96, 128, 16, 32, 32, 'inception_3a')
            block_3b = inception_module(block_3a, 128, 128, 192, 32, 96, 64, 'inception_3b')
            pool3 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding='SAME', name='MaxPool_3')(block_3b)

            block_4a = inception_module(pool3, 192, 96, 208, 16, 48, 64, 'inception_4a')
            block_4b = inception_module(block_4a, 160, 112, 224, 24, 64, 64, 'inception_4b')
            block_4c = inception_module(block_4b, 128, 128, 256, 24, 64, 64, 'inception_4c')
            block_4d = inception_module(block_4c, 112, 144, 288, 32, 64, 64, 'inception_4d')
            block_4e = inception_module(block_4d, 256, 160, 320, 32, 128, 128, 'inception_4e')
            pool4 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding='SAME', name='MaxPool_4')(block_4e)

            block_5a = inception_module(pool4, 256, 160, 320, 32, 128, 128, 'inception_5a')
            block_5b = inception_module(block_5a, 384, 192, 384, 48, 128, 128, 'inception_5b')

    def attach_loss(self):
        with self.graph.as_default():
            labels = tf.placeholder(tf.int64, shape=(None,), name='labels')

            with tf.variable_scope('losses'):
                main_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
                aux_4a_loss = tf.losses.sparse_softmax_cross_entropy(labels, aux_logit_4a)
                aux_4d_loss = tf.losses.sparse_softmax_cross_entropy(labels, aux_logit_4d)
                loss = main_loss + 0.3 * aux_logit_4a + 0.3 * aux_4d_loss


    with graph.as_default():
        lr = tf.placeholder_with_default(1e-2, (), name='learning_rate')
        train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)

    def attach_metric(self):
        with graph.as_default():
            with tf.variable_scope('metrics'):
                top_5, top_5_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(logits, labels, k=5), tf.float32) * 100)

                top_1, top_1_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32) * 100)

                metric_loss, metric_loss_op = tf.metrics.mean(main_loss)

                metric_init_op = tf.group([var.initializer for var in graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)],
                                          name='metric_init_op')
                metric_update_op = tf.group([top_5_op, top_1_op, metric_loss_op], name='metric_update_op')

                top_5 = tf.identity(top_5, 'top5_acc')
                top_1 = tf.identity(top_1, 'top1_acc')
                metric_loss = tf.identity(metric_loss, 'metric_loss')

                tf.summary.scalar('top5_accuracy', top_5)
                tf.summary.scalar('top1_accuracy', top_1)
                tf.summary.scalar('loss', metric_loss)
                merged = tf.summary.merge_all()
