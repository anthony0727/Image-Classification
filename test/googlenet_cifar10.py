import tensorflow as tf
from tqdm import tqdm

from googlenet import GoogLeNet

net = GoogLeNet()
net.build((32, 32, 3), 100)
net.train_writer.add_graph(net.graph)

from data import Cifar, generator, images_augmentation

num_epoch = 100
num_batch = 128

cifar = Cifar(100)
gen = generator(cifar.x_train, cifar.y_train, num_batch)

sess = tf.Session(graph=net.graph)
print(cifar.x_train.shape)
with net.graph.as_default():
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.MomentumOptimizer(net.lr, 0.9).minimize(net.loss, global_step=global_step)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    for epoch in range(num_epoch):
        for step in tqdm(range(len(cifar.x_train) // num_batch)):
            batch_x, batch_y = next(gen)
            print(batch_x.shape)
            batch_x = sess.run(images_augmentation(batch_x, is_train=net.is_train))

            sess.run(train_op, feed_dict={net.xs: batch_x, net.ys: batch_y, net.lr: 0.01, net.is_train: True})
