# -*- coding: utf-8 -*-

from model.ABCNet import *

he_init = tf.initializers.he_normal()
xavier_init = tf.initializers.glorot_normal()


def residual_block(input_layer, is_train, filters, strides=(1, 1), block_name="residual_block"):
    with tf.variable_scope(block_name):
        if input_layer.shape.as_list()[-1] != filters:
            # input_layer의 필터 갯수와 filters가 다르면, projection layer을 거침
            projection = tf.layers.Conv2D(filters, (1, 1), strides=strides)(input_layer)
        else:
            # 동일하면 바로 이어줌
            projection = input_layer

        conv = tf.layers.Conv2D(filters, (3, 3), strides, padding='SAME',
                                kernel_initializer=he_init)(input_layer)
        bn = tf.layers.BatchNormalization()(conv, training=is_train)
        act = tf.nn.relu(bn)
        conv = tf.layers.Conv2D(filters, (3, 3), padding='SAME',
                                kernel_initializer=he_init)(act)
        bn = tf.layers.BatchNormalization()(conv, training=is_train)
        added = tf.add(projection, bn)
        out = tf.nn.relu(added)

    return out


class ResNet(Network):
    def __init__(self):
        # input_shape = (224, 224, 3)
        # num_classes = 1000
        self.is_train = tf.placeholder_with_default(False, (), name='is_train')

    def attach_placeholders(self):
        self.xs = tf.placeholder(tf.float32, (None, *self.image_shape), name='xs')
        self.ys = tf.placeholder(tf.int32, (None,), name='labels')

        with tf.variable_scope("preprocess"):
            image_mean = tf.constant([123.68, 116.779, 103.939], tf.float32)
            self.xs = self.xs - image_mean

    def attach_layers(self):
        layer = tf.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='SAME', name='conv1')(self.xs)
        layer = tf.layers.MaxPooling2D((3, 3), (2, 2), padding='SAME', name='maxpool1')(layer)

        block_names = ['conv2_1', 'conv2_2', 'conv2_3']
        for block_name in block_names:
            layer = residual_block(layer, self.is_train, 64, block_name=block_name)

        block_names = ['conv3_2', 'conv3_3', 'conv3_4']
        layer = residual_block(layer, self.is_train, 128, strides=(2, 2), block_name='conv3_1')
        for block_name in block_names:
            layer = residual_block(layer, self.is_train, 128, block_name=block_name)

        layer = residual_block(layer, self.is_train, 256, strides=(2, 2), block_name='conv4_1')
        block_names = ['conv4_2', 'conv4_3', 'conv4_4', 'conv4_5', 'conv4_6']
        for block_name in block_names:
            layer = residual_block(layer, self.is_train, 256, block_name=block_name)

        layer = residual_block(layer, self.is_train, 512, strides=(2, 2), block_name='conv4_1')
        block_names = ['conv5_2', 'conv5_3']
        for block_name in block_names:
            layer = residual_block(layer, self.is_train, 512, block_name=block_name)

        gap = tf.reduce_mean(layer, axis=(1, 2), name='global_average_pooling')
        self.logits = tf.layers.Dense(self.n_class, kernel_initializer=xavier_init, name='logits')(gap)
        self.y_pred = tf.nn.softmax(self.logits, name='prediction')

    def attach_loss(self):
        pass

    def attach_metric(self):
        pass

    def attach_summary(self):
        pass
