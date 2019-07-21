import os

import tensorflow as tf
from tqdm import tqdm
from util.data_helper import augment_images


# move to main
# train_set = Dataset(train_x, train_y)
# test_set = Dataset(test_x, test_y)

def feed(x, y, is_train=False):
    feed_dict = {
        'xs:0': x,
        'ys:0': y,
        'is_train:0': is_train
    }

    return feed_dict


class Trainer:
    def __init__(self, sess, train_set, test_set, log_dir, n_epoch=100, n_batch=128):
        self.sess = sess
        self.train_set = train_set
        self.test_set = test_set
        self.log_dir = log_dir
        self.test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, './test'))
        self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, './train'), self.sess.graph)

        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.n_data = len(train_set)

    def run(self):
        sess = self.sess
        graph = self.sess.graph

        with graph.as_default():
            metric_init_op = graph.get_operation_by_name('metrics/metric_init_op')
            metric_update_op = graph.get_operation_by_name('metrics/metric_update_op')

            top1 = graph.get_tensor_by_name('metrics/top1_accuracy:0')
            top5 = graph.get_tensor_by_name('metrics/top5_accuracy:0')

            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])

            global_step = tf.train.get_global_step(graph)
            for epoch in range(self.n_epoch):
                for step in tqdm(range(self.n_data // self.n_batch)):
                    batch_x, batch_y = self.train_set.next_batch(self.n_batch)
                    batch_x = augment_images(batch_x)
                    sess.run(tf.get_collection(tf.GraphKeys.TRAIN_OP),
                             feed_dict=feed(batch_x, batch_y, True))

                self.train_set.shuffle()

                sess.run(metric_init_op)
                for step in range(0, len(self.train_set) // 1000):
                    batch_x = self.train_set.images[step * 1000:(step + 1) * 1000]
                    batch_y = self.train_set.labels[step * 1000:(step + 1) * 1000].ravel()

                    sess.run(metric_update_op,
                             feed_dict=feed(batch_x, batch_y))

                summary, top1_val, top5_val = self.sess.run(
                    tf.get_collection(tf.GraphKeys.SUMMARIES) + [top1, top5],
                    feed_dict=feed(batch_x, batch_y))

                print('[{:3d} epoch train top-1 acc : {:2.2f}% | top-5 acc : {:2.2f}%' \
                      .format(epoch, top1_val, top5_val))
                self.train_writer.add_summary(summary, global_step.eval(sess))

                sess.run(metric_init_op)
                for step in range(0, len(self.test_set) // 1000):
                    batch_x, self.test_set.images[step * 1000:(step + 1) * 1000]
                    batch_y, self.test_set_labels[step * 1000:(step + 1) * 1000].ravel()

                    sess.run(metric_update_op,
                             feed_dict=feed(batch_x, batch_y))

                summary, top1_val, top5_val = sess.run(
                    tf.get_collection(tf.GraphKeys.SUMMARIES) + [top1, top5],
                    feed_dict=feed(batch_x, batch_y))

                print('[{:3d} epoch test top-1 acc : {:2.2f}% | top-5 acc : {:2.2f}%' \
                      .format(epoch, top1_val, top5_val))

                self.test_writer.add_summary(summary, global_step.eval(sess))
