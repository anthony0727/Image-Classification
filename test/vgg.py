import gc
import os
import shutil

import tensorflow as tf
from keras.datasets.cifar100 import load_data
from tqdm import tqdm

from data_helper import Dataset, augment_images
from train_helper import Trainer

input_shape = (None, 32, 32, 3)
n_class = 100

graph = tf.Graph()
with graph.as_default():
    images = tf.placeholder(tf.float32, input_shape, name='images')
    labels = tf.placeholder(tf.int32, (None,), name='labels')
    is_train = tf.placeholder_with_default(False, (), name='is_train')

    with tf.variable_scope('preprocss'):
        vgg_mean = tf.constant([123.68, 116.779, 103.939], tf.float32)
        x = images - vgg_mean

    with tf.variable_scope('VGGBlock-1'):
        conv = tf.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, name='conv1')(x)

    with tf.variable_scope('VGGBlock-2'):
        conv = tf.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu, name='conv1')(conv)

    with tf.variable_scope('VGGBlock-3'):
        conv = tf.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu, name='conv1')(conv)
        conv = tf.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu, name='conv2')(conv)
    pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='MaxPool-3')(conv)

    with tf.variable_scope('VGGBlock-4'):
        conv = tf.layers.Conv2D(256, (3, 3), padding='same', activation=tf.nn.relu, name='conv1')(pool)
        conv = tf.layers.Conv2D(256, (3, 3), padding='same', activation=tf.nn.relu, name='conv2')(conv)
    pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='MaxPool-4')(conv)

    with tf.variable_scope('VGGBlock-5'):
        conv = tf.layers.Conv2D(256, (3, 3), padding='same', activation=tf.nn.relu, name='conv1')(pool)
        conv = tf.layers.Conv2D(256, (3, 3), padding='same', activation=tf.nn.relu, name='conv2')(conv)
    pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='MaxPool-5')(conv)

    with tf.variable_scope('FC'):
        fc = tf.layers.Flatten()(pool)

        fc = tf.layers.Dense(1024, activation=tf.nn.relu)(fc)
        dropout = tf.layers.Dropout(0.5)(fc, training=is_train)
        fc = tf.layers.Dense(1024, activation=tf.nn.relu)(dropout)
        dropout = tf.layers.Dropout(0.5)(fc, training=is_train)
        logits = tf.layers.Dense(n_class)(dropout)

    logits = tf.identity(logits, name='logits')
    y_pred = tf.nn.softmax(logits, name='predictions')

weight_decay = 5e-4

with graph.as_default():
    with tf.variable_scope('losses'):
        sce = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.global_variables()])
        loss = sce + weight_decay * l2_loss

    loss = tf.identity(loss, name='loss')

with graph.as_default():
    with tf.variable_scope('metrics'):
        top5, top5_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(y_pred, labels, k=5), tf.float32) * 100)
        top1, top1_op = tf.metrics.mean(tf.cast(tf.nn.in_top_k(y_pred, labels, k=1), tf.float32) * 100)
        metric_loss, loss_op = tf.metrics.mean(loss)

        metric_init_op = tf.group(
            [var.initializer for var in graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)],
            name='metric_init_op')
        metric_update_op = tf.group([top5_op, top1_op, loss_op], name='metric_update_op')

        top5 = tf.identity(top5, 'top5_accuracy')
        top1 = tf.identity(top1, 'top1_accuracy')
        tf.identity(metric_loss, 'metric_loss')

        tf.summary.scalar('top5_tb', top5)
        tf.summary.scalar('top1_tb', top1)
        tf.summary.scalar('loss_tb', metric_loss)
        merged = tf.summary.merge_all()

if __name__ == '__main__':

    with graph.as_default():
        lr = tf.placeholder_with_default(1e-4, (), name='learning_rate')
        global_step = tf.train.get_or_create_global_step()
        momentum = 0.9
        with tf.variable_scope('optimizer'):
            train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step)

    LOG_DIR = './log'
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    os.makedirs(LOG_DIR, exist_ok=True)

    (train_x, train_y), (test_x, test_y) = load_data()

    train_y = train_y.reshape((-1,))
    test_y = test_y.reshape((-1,))

    train_set = Dataset(train_x, train_y)
    test_set = Dataset(test_x, test_y)

    sess = tf.Session(graph=graph)

    trainer = Trainer(pretrained_vgg, train_set, test_set, LOG_DIR)
    pretrained_vgg = trainer.run()

    vgg.transfer(pretrained_vgg)

    del trainer, pretrained_vgg
    gc.collect()

    trainer = Trainer(vgg, train_set, test_set, LOG_DIR)




