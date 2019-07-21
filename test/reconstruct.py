import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir = './inception_model'
path_pb = 'classify_image_graph_def.pb'
path = os.path.join(data_dir, path_pb)
graph = tf.Graph()

with graph.as_default():
    with tf.gfile.FastGFile(path, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')

        y_pred = graph.get_tensor_by_name('softmax:0')
        y_logits = graph.get_tensor_by_name('softmax/logits:0')
        resized_image = graph.get_tensor_by_name('ResizeBilinear:0')
        transfer_layer = graph.get_tensor_by_name('pool_3:0')

        # Filter Visualization
        # weights = graph.get_tensor_by_name('mixed_8/join:0')
        weights = graph.get_tensor_by_name('conv_2/Conv2D:0')
        pattern_loss = tf.reduce_mean(weights[:, :, :, 0])
        pattern_grad = tf.gradients(pattern_loss, resized_image)[0]

        sess = tf.Session(graph=graph)

    if not os.path.isdir('./reconstructor_log'):
        writer = tf.summary.FileWriter('./reconstructor_log')
        writer.add_graph(graph)

image = np.random.uniform(size=resized_image.get_shape()) + 128.0 # 128 is mean value

with graph.as_default():
    for i in range(100):
        feed_dict = {resized_image: image}
        grad, loss_value = sess.run([pattern_grad, pattern_loss], feed_dict=feed_dict)
        grad = np.array(grad).squeeze()
        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad # gradient ascent
        image = np.clip(image, 0.0, 255.0)

    image = (image[0] - image[0].min()) / (image[0].max() - image[0].min())
    plt.imshow(image)
    plt.show()

