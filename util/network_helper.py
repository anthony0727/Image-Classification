import tensorflow as tf


def dense_bn_relu(xs, is_train, filters, name='dense'):
    with tf.variable_scope(name):
        dense = tf.layers.Dense(filters, use_bias=False)(xs)
        bn = low_level_batch_normalize(dense, is_train)

    return tf.nn.relu(bn)


def low_level_batch_normalize(xs, is_train):
    decay = 0.999

    with tf.variable_scope('normalization'):
        epsilon = tf.constant(1e-3)

        input_size = xs.get_shape()[-1]
        gamma = tf.Variable(tf.one(input_size), name='scale_factor')
        beta = tf.Variable(tf.zeros(input_size), name='shift_factor')

        train_mean = tf.Variable(tf.zeros(input_size), trainable=False)
        train_var = tf.Variable(tf.zeros(input_size), trainable=False)

    def train_phase():
        batch_mean, batch_var = tf.nn.moments(xs, axes=[0])
        update_mean = tf.assign(train_mean,
                                train_mean * decay + batch_mean * (1 - decay))

        update_var = tf.assign(train_var,
                               train_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([update_mean, update_var]):
            x_norm = (xs - batch_mean) / tf.sqrt(batch_var + epsilon)
            ys = gamma * x_norm + beta

        return ys

    def test_phase():
        x_norm = (xs - train_mean) / tf.sqrt(train_var + epsilon)
        ys = gamma * x_norm + beta

        return ys

    ys = tf.cond(is_train, train_phase, test_phase)

    return ys
