"""
this file is to be presented in .ipynb format
"""

import gc

import os
from model.vggnet import VGGNet
from data import Cifar, generator

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
    return network


if __name__ == '__main__':
    data = Cifar10()  # data class

    net = VGGNet()

    pretrained_net = train(net, data)
    pretrained_net.show()  # tensorboard

    reconstructed_net = VGGNet()

    reconstructed_net.transfer(pretrained_net)

    del pretrained_net
    gc.collect()

    reconstructed_net = train(reconstructed_net, data, 'adam')
    reconstructed_net.show()  # tensorboard
