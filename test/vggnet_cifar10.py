# Load Module

import tensorflow as tf
import gc

from model.vgg import VGG
from util import train

# Declare Static Variables

log_dir = './log'

input_shape = (32, 32, 3)
n_class = 100

# Build VGG11

pretrained_vgg = VGG(11)
pretrained_vgg.build(input_shape, n_class)

# Pretrain VGG11

with pretrained_vgg.graph.as_default() as graph:
    loss = graph.get_tensor_by_name('loss:0')
    lr = tf.placeholder_with_default(1e-2, (), name='learning_rate')
    global_step = tf.train.get_or_create_global_step()

    with tf.variable_scope('optimizer'):
        tf.train.MomentumOptimizer(lr, 0.9).minimize(loss, global_step)

    sess = tf.Session(graph=graph)
    sess = train(sess, log_dir)
    saver = tf.train.Saver()
    saver.save(sess, './vgg/vgg11')

# Reserve some memory

del pretrained_vgg.graph
gc.collect()

# Build VGG13

reconstructed_vgg = VGG(13)
reconstructed_vgg.build(input_shape, n_class)

# Collect weights to transfer

with reconstructed_vgg.graph.as_default():
    transfer_weights = \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '(VGGBLOCK-1/conv1)') + \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '(VGGBLOCK-2/conv1)') + \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '(VGGBLOCK-3/conv1|conv2)') + \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FC/')

# Restore pretrained weights

with reconstructed_vgg.graph.as_default() as graph:
    sess = tf.Session(graph=graph)

    saver = tf.train.Saver(var_list=transfer_weights)
    saver.restore(sess, './vgg/vgg11')

# Train VGG13

with reconstructed_vgg.graph.as_default() as graph:
    loss = graph.get_tensor_by_name('loss:0')
    lr = tf.placeholder_with_default(1e-2, (), name='learning_rate')
    global_step = tf.train.get_or_create_global_step()

    with tf.variable_scope('optimizer'):
        tf.train.AdamOptimizer(lr).minimize(loss, global_step)

    with tf.variable_scope('initialization'):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])

        uninitialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(uninitialized_vars):
        sess.run(tf.variables_initializer(uninitialized_vars))

    sess = tf.Session(graph=graph)
    sess = train(sess, log_dir)
    saver = tf.train.Saver()
    saver.save(sess, './vgg/vgg11')


