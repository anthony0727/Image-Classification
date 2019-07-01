import tensorflow as tf
import pandas as pd
import numpy as np
import numpy.random as npr
from tqdm import tqdm

from model.ABCNet import Network

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

        out4 = tf.layers.MaxPooling2D((3, 3), (1, 1), padding='SAME', name='3x3_pool')(prev_layer)
        out4 = tf.layers.Conv2D(after_pool_filters, (5, 5), padding='SAME', kernel_initializer=he_init,
                                activation=tf.nn.relu, name='after_pool')(out4)

        out = tf.concat([out1, out2, out3, out4], axis=-1, name='filter_concatenation')

    return out


class GoogLeNet(Network):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        # input_shape = (224, 224, 3)
        # num_classes = 1000
        self.input_shape = None

        # additional attrs
        self.top5_accuracy, self.top1_accuracy = None, None
        self.aux_4a_loss, self.aux_4d_loss = None, None
        self.aux_logit_4a, self.aux_logit_4d = None, None

    def attach_placeholders(self):
        images = tf.placeholder(tf.float32, (None, *self.input_shape), name='images')
        image_mean = tf.constant([123.68, 116.779, 103.939], tf.float32)

        self.xs = images - image_mean
        self.ys = tf.placeholder(tf.int32, (None,), name='labels')
        self.lr = tf.placeholder_with_default(1e-2, (), name='learning_rate')

    def auxiliary_network(self, block, name):
        with tf.variable_scope('auxiliary_network_' + name):
            # avg_pool = tf.layers.AveragePooling2D((5, 5), (3, 3))(block)
            conv = tf.layers.Conv2D(64, (1, 1), kernel_initializer=he_init, activation=tf.nn.relu, name='1x1')(
                block)

            fc = tf.layers.Flatten()(conv)
            fc = tf.layers.Dense(512, kernel_initializer=he_init, activation=tf.nn.relu)(fc)
            fc = tf.layers.Dropout(0.7)(fc)
            aux_logit = tf.layers.Dense(self.n_class, kernel_initializer=xavier_init)(fc)

        return aux_logit

    def attach_layers(self):
        he_init = tf.initializers.he_uniform()

        conv1 = tf.layers.Conv2D(32, (3, 3), (1, 1),
                                 padding='SAME', kernel_initializer=he_init, name='3x3_conv_1')(self.xs)
        pool1 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MaxPool_1')(conv1)

        conv2 = tf.layers.Conv2D(96, (3, 3), padding='SAME', kernel_initializer=he_init, name='3x3_conv_2')(pool1)
        pool2 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MaxPool_2')(conv2)

        block_3a = inception_module(pool2, 32, 48, 64, 8, 16, 16, 'inception_3a')
        block_3b = inception_module(block_3a, 64, 64, 96, 16, 48, 32, 'inception_3b')
        pool3 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding='SAME', name='MaxPool_3')(block_3b)

        block_4a = inception_module(pool3, 96, 48, 104, 8, 24, 32, 'inception_4a')
        block_4b = inception_module(block_4a, 80, 56, 112, 12, 32, 32, 'inception_4b')
        block_4c = inception_module(block_4b, 64, 64, 128, 12, 32, 32, 'inception_4c')
        block_4d = inception_module(block_4c, 56, 72, 144, 16, 32, 32, 'inception_4d')
        block_4e = inception_module(block_4d, 128, 80, 160, 16, 64, 64, 'inception_4e')
        pool4 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding='SAME', name='MaxPool_4')(block_4e)

        block_5a = inception_module(pool4, 128, 80, 160, 16, 64, 64, 'inception_5a')
        block_5b = inception_module(block_5a, 192, 96, 192, 24, 64, 64, 'inception_5b')

        layer = tf.layers.Flatten()(block_5b)
        layer = tf.layers.Dropout(0.4)(layer, training=self.is_train)
        layer = tf.layers.Dense(512, kernel_initializer=he_init, activation=tf.nn.relu)(layer)
        self.logits = tf.layers.Dense(self.n_class, kernel_initializer=xavier_init, name='logits')(layer)
        y_pred = tf.nn.softmax(self.logits)

        self.aux_logit_4a, self.aux_logit_4d = \
            self.auxiliary_network(block_4a, '4a'), self.auxiliary_network(block_4d, '4d')

    def attach_loss(self):
        with tf.variable_scope('losses'):
            loss = tf.losses.sparse_softmax_cross_entropy(self.ys, self.logits)
            self.aux_4a_loss = tf.losses.sparse_softmax_cross_entropy(self.ys, self.aux_logit_4a)
            self.aux_4d_loss = tf.losses.sparse_softmax_cross_entropy(self.ys, self.aux_logit_4d)
            self.loss = loss + 0.3 * self.aux_4a_loss + 0.3 * self.aux_4d_loss

    def attach_metric(self):
        with tf.variable_scope('metrics'):
            top5, top5_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(self.logits, self.ys, k=5), tf.float32) * 100)

            top1, top1_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(self.logits, self.ys, k=1), tf.float32) * 100)

            metric_loss, metric_loss_op = tf.metrics.mean(self.loss)

            metric_init_op = tf.group(
                [var.initializer for var in self.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)],
                name='metric_init_op')
            metric_update_op = tf.group([top5_op, top1_op, metric_loss_op], name='metric_update_op')

            self.top5_accuracy = tf.identity(top5, 'top5_acc')
            self.top1_accuracy = tf.identity(top1, 'top1_acc')
            tf.identity(metric_loss, 'metric_loss')

    def attach_summary(self):
        # for metric in self.metrics:
        #     tf.summary.scalar(*metric)
        tf.summary.scalar('top5_accuracy', self.top5_accuracy)
        tf.summary.scalar('top1_accuracy', self.top1_accuracy)
        tf.summary.scalar('loss', self.loss)
        self.merge_all = tf.summary.merge_all(name='merge_all')

    def transfer(self):
        pass