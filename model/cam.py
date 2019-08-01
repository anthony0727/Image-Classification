# -*- coding: utf-8 -*-

# import os
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# data_dir = './inception_model'
# path_pb = 'classify_image_graph_def.pb'
# path = os.path.join(data_dir, path_pb)
# graph = tf.Graph()
#
# with graph.as_default() as graph:
#     with tf.gfile.FastGFile(path, 'rb') as file:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(file.read())
#         tf.import_graph_def(graph_def, name='')
#
#         pool_3 = graph.get_tensor_by_name('')
#
#         xs = graph.get_tensor_by_name('DecodeJpeg:0')
#         dense_weights = graph.get_tensor_by_name('dense/kernel:0')
#         classmap = graph.get_tensor_by_name('')
#         phase_train = graph.get_tensor_by_name('phase_train:0')
#
#         images = []
#
#         label_num = tf.placeholder(tf.float32)
#
#         sliced_dense = dense_weights[:, label_num:label_num + 1]
#         tf.image.resize_bilinear(pool_3, size=[100, 100])
#
#         classmap = (classmap - tf.reduce_min(classmap, axis=[0, 1])) / \
#                    (tf.reduce_max(classmap, axis=[0, 1]) - tf.reduce_min(classmap, axis=[0, 1]))
#
#         index = 10
#         sess = tf.Session()
#         act_images = sess.run(classmap, feed_dict={xs: images[index:index + 1], phase_train: phase_train})
#         act_image = act_images[:, :, 0]
#
#         plt.imshow(act_image_)
#         ori = images[index]
#
#         cmap = cv2.applyColorMap(np.uint32(act_image * 255), cv2.COLORMAP_JET)
#         cmap = cv2.cvtColor(cmap_, cv2.COLOR_BGR2RGB)
#         ori = np.uint8(ori * 255)
#
#         plt.imshow(act_image_)
#         plt.show()
#
#         plt.imshow(cmap)
#         plt.show()
#
#         plt.imshow(ori)
#         plt.show()
#         images = cv2.addWeighted(ori, 0.5, cmap, 0.3, 0)
#         plt.imshow(images)
#         plt.show()
#
#
# # image stitching 20 images
#
# stitch_images = []
# for i in range(20):
#     stitch_images.append()
#     pass
#
#
# merged_image = np.stack(stitch_images)
# merged_image.reshape((5,4,32,32,3).transpose([0,2,1,3,4]).reshape([5*32, 4*32, 3])