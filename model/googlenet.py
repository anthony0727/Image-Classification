import tensorflow as tf
import pandas as pd
import numpy as np
import numpy.random as npr

from ABCNet import Network

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


def auxiliary_network(block, name):
    with tf.variable_scope('auxiliary_network_' + name):
        avg_pool = tf.layers.AveragePooling2D((5, 5), (3, 3))(block)
        conv = tf.layers.Conv2D(128, (1, 1), kernel_initializer=he_init, activation=tf.nn.relu, name='1x1')(avg_pool)

        fc = tf.layers.Flatten()(conv)
        fc = tf.layers.Dense(1024, kernel_initializer=he_init, activation=tf.nn.relu)(fc)
        fc = tf.layers.Dropout(0.7)(fc)
        aux_logit = tf.layers.Dense(1000, kernel_initializer=xavier_init)(fc)

    return aux_logit


class GoogLeNet(Network):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        # input_shape = (224, 224, 3)
        # num_classes = 1000
        self.input_shape = None
        self.num_class = 0

        # additional attrs
        self.top5_accuracy, self.top1_accuracy = None, None
        self.aux_4a_loss, self.aux_4d_loss = None, None
        self.aux_logit_4a, self.aux_logit_4d = None, None


    def attach_placeholders(self):
        with self.graph.as_default():
            images = tf.placeholder(tf.float32, (None, *self.input_shape), name='images')
            image_mean = tf.constant([123.68, 116.779, 103.939])

            self.xs = images - image_mean
            self.ys = tf.placeholder(tf.int32, (None,), name='labels')
            self.lr = tf.placeholder_with_default(1e-2, (), name='learning_rate')

    def train(self, optimizer):
        pass

    def transfer(self):
        pass

    def attach_layers(self):
        with self.graph.as_default():
            he_init = tf.initializers.he_uniform()

            conv1 = tf.layers.Conv2D(64, (7, 7), (2, 2), padding='SAME', kernel_initializer=he_init, name='7x7_conv')(x)
            pool1 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MaxPool_1')(conv1)
            conv2 = tf.layers.Conv2D(192, (3, 3), padding='SAME', kernel_initializer=he_init, name='3x3_conv')(pool1)

            pool2 = tf.layers.MaxPooling2D((3, 3), (2, 2), name='MaxPool_2')(conv2)

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

            self.aux_logit_4a, self.aux_logit_4d = auxiliary_network(block_4a, '4a'), auxiliary_network(block_4d, '4d')

    def attach_loss(self):
        with self.graph.as_default():
            with tf.variable_scope('losses'):
                loss = tf.losses.sparse_softmax_cross_entropy(self.ys, self.logits)
                self.aux_4a_loss = tf.losses.sparse_softmax_cross_entropy(self.ys, self.aux_logit_4a)
                self.aux_4d_loss = tf.losses.sparse_softmax_cross_entropy(self.ys, self.aux_logit_4d)
                self.loss = loss + 0.3 * self.aux_4a_loss + 0.3 * self.aux_4d_loss

    def attach_metric(self):
        with self.graph.as_default():
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
        with self.graph.as_default():
            # for metric in self.metrics:
            #     tf.summary.scalar(*metric)
            tf.summary.scalar('top5_accuracy', self.top5_accuracy)
            tf.summary.scalar('top1_accuracy', self.top1_accuracy)
            tf.summary.scalar('loss', self.loss)
        tf.summary.merge_all(name='merge_all')
