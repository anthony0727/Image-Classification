"""
this file is to be presented in .ipynb format
"""

import gc

import tensorflow as tf

from util.data_helper import load_cifar100, Dataset
from model.vggnet import VGGNet
from util.train_helper import Trainer

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer
}

# hparam
batch_size = 100
min_loss = 1000000.0
learning_rate = 0.0005
epochs = 10000

LOG_DIR = './log'

input_shape = (32, 32, 3)
n_class = 100

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = load_cifar100()

    train_set = Dataset(train_x, train_y)
    test_set = Dataset(test_x, test_y)

    prms = (input_shape, n_class)

    net = VGGNet(11)
    net.build(*prms)

    sess = tf.Session(graph=net.graph)
    with net.graph.as_default():
        trnr = Trainer(sess, train_set, test_set, LOG_DIR)
        trnr.n_epoch = 1
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        pretrained_net = trnr.run()
        # pretrained_net.show() tensorboard

        reconstructed_net = VGGNet(13)
        reconstructed_net.build(*prms)

        transfer_ops = reconstructed_net.transfer_ops(pretrained_net)

    with reconstructed_net.graph.as_default():
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run(transfer_ops)

    del pretrained_net, sess
    gc.collect()

    sess = tf.Session(graph=reconstructed_net.graph)
    reconstructed_net = Trainer(sess, train_set, test_set, LOG_DIR).run()
    # reconstructed_net.show() tensorboard
