import os

import tensorflow as tf
from tqdm import tqdm
from util.data_helper import load_cifar100, augment_images, Dataset


def feed(x, y, is_train=False):
    feed_dict = {
        'xs:0': x,
        'ys:0': y,
        'is_train:0': is_train
    }

    return feed_dict


def train(sess, log_dir, n_epoch=128, n_batch=128):
    graph = sess.graph

    (train_x, train_y), (test_x, test_y) = load_cifar100()

    train_set = Dataset(train_x, train_y, n_batch=n_batch)
    test_set = Dataset(test_x, test_y, n_batch=n_batch)

    n_data = len(train_set)

    sess = tf.Session(graph=graph)
    with graph.as_default():
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    with graph.as_default():
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, './train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, './test'))

        metric_init_op = tf.group([var.initializer for var in graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)])
        metric_update_op = graph.get_operation_by_name('metric/update_op')

        top1 = graph.get_tensor_by_name('metric/top1_accuracy:0')
        top5 = graph.get_tensor_by_name('metric/top5_accuracy:0')

        for epoch in range(n_epoch):
            for step in tqdm(range(n_data // n_batch)):
                batch_x, batch_y = train_set.next_batch()
                batch_x = augment_images(batch_x)
                sess.run(tf.get_collection(tf.GraphKeys.TRAIN_OP),
                         feed_dict=feed(batch_x, batch_y, True))

            train_set.shuffle()

            sess.run(metric_init_op)
            for step in range(0, len(train_set) // 1000):
                batch_x = train_set.images[step * 1000:(step + 1) * 1000]
                batch_y = train_set.labels[step * 1000:(step + 1) * 1000].ravel()

                sess.run(metric_update_op,
                         feed_dict=feed(batch_x, batch_y))

            *summaries, top1_val, top5_val = sess.run(
                tf.get_collection(tf.GraphKeys.SUMMARIES) + [top1, top5],
                feed_dict=feed(batch_x, batch_y))

            print('[{:3d} epoch train top-1 acc : {:2.2f}% | top-5 acc : {:2.2f}%' \
                  .format(epoch, top1_val, top5_val))
            for summary in summaries:
                train_writer.add_summary(summary, step)

            sess.run(metric_init_op)
            for step in range(0, len(test_set) // 1000):
                batch_x, test_set.images[step * 1000:(step + 1) * 1000]
                batch_y, test_set.labels[step * 1000:(step + 1) * 1000].ravel()

                sess.run(metric_update_op,
                         feed_dict=feed(batch_x, batch_y))

            *summaries, top1_val, top5_val = sess.run(
                tf.get_collection(tf.GraphKeys.SUMMARIES) + [top1, top5],
                feed_dict=feed(batch_x, batch_y))

            print('[{:3d} epoch test top-1 acc : {:2.2f}% | top-5 acc : {:2.2f}%' \
                  .format(epoch, top1_val, top5_val))

            for summary in summaries:
                test_writer.add_summary(summary, step)

    return sess
