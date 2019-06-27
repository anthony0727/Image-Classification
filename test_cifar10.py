"""
this file is to be presented in .ipynb format
"""

import gc

import os
from vggnet import VGGNet
from data import Cifar10

import tensorflow as tf
import numpy as np
import numpy.random as npr

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer
}

# hparam
batch_size = 100
min_loss = 1000000.0
learning_rate = 0.0005
epochs = 10000

ROOT = '.'


def train(network, data, optimizer='momentum'):
    graph = network.graph

    with tf.Session(graph) as sess:
        lr, loss, acc, merged = graph.get_tensor_by_name('lr:0'), \
                                graph.get_tensor_by_name('loss:0'), \
                                graph.get_tensor_by_name('accuracy:0'), \
                                graph.get_tensor_by_name('merge_all:0')

        optimizer = OPTIMIZERS[optimizer]
        train_op = optimizer(lr).minimize(loss)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        save_root_folder = os.path.join(ROOT, 'save_root_folder')

        train_generator = Cifar10.generator(data.x_train, data.y_train, batch_size)
        for i in range(epochs):
            batch_xs, batch_ys = next(train_generator)
            feeds = {'xs': batch_xs, 'ys': batch_ys, 'lr': learning_rate}
            # training
            _ = network.sess.run(train_op, feed_dict=feeds)

            if i % 100 == 0:
                # Validate validation dataset
                fetches = [loss, acc, merged]
                val_loss, val_acc, val_merged = sess.run(fetches)

                # Validate train dataset : extract randomly 10000 samples from train dataset
                batch_xs, batch_ys = npr.choice(data.x_train, 10000), npr.choice(data.y_train, 10000)

                train_loss, train_acc, train_merged = sess.run(merged)

                print('step : {} train loss : {:.4f} acc : {:.4f} | Val loss : {:.4f} acc : {:.4f}'. \
                      format(i, train_loss, train_acc, val_loss, val_acc))

                # Save Model
                min_loss = min(val_loss, min_loss)
                if val_loss < min_loss:
                    min_loss = val_loss
                    # save_path =  # fix me #
                    # graph.saver.save(  # fix me#)
                    #     print('model save!')
                    #
                    # # Add values to tensorboard
                    # graph.train_writer.add_summary()
                    # graph.test_writer.add_summary()
                    # graph.train_writer.flush()
    return network


if __name__ == '__main__':
    data = Cifar10()  # data class

    net = VGGNet()

    pretrained_net = net.fit(data)
    pretrained_net.show()  # tensorboard

    reconstructed_net = VGGNet()

    reconstructed_net.transfer(pretrained_net, data)

    del pretrained_net
    gc.collect()

    reconstructed_net = train(reconstructed_net(data, 'adam'))
    reconstructed_net.show()  # tensorboard
