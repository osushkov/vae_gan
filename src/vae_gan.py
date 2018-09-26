
import data_provider
import matplotlib.pyplot as plt
import numpy as np
import random
import sonnet as snt
import tensorflow as tf


def _showRandom(data_source):
    img, loc, label = data_source()


    print("loc: {}".format(loc))
    print("label: {}".format(label))

    plt.imshow(img)
    plt.show()


data_source = data_provider.DataProvider(48, 48)
_showRandom(data_source)
training_data = []
for _ in range(1000):
    training_data.append(data_source())

training_images = np.vstack([np.expand_dims(d[0], 0) for d in training_data])
training_locs = np.vstack([np.expand_dims(d[1], 0) for d in training_data])
training_labels = np.vstack([np.expand_dims(d[2], 0) for d in training_data])

print("shapes: {} {} {}".format(training_images.shape,
                                training_locs.shape,
                                training_labels.shape))

images_dataset = tf.data.Dataset.from_tensor_slices(training_images)
locs_dataset = tf.data.Dataset.from_tensor_slices(training_locs)
labels_dataset = tf.data.Dataset.from_tensor_slices(training_labels)

dataset = tf.data.Dataset.zip((images_dataset, locs_dataset, labels_dataset))
dataset = dataset.batch(32)

print("output types: {} {}".format(dataset.output_types, dataset.output_shapes))

value = dataset.make_one_shot_iterator().get_next()
# sample_op = tf.py_func(data_source, [], [tf.float32, tf.float32, tf.float32])
# queue = tf.FIFOQueue(capacity=128, dtype=tf.float32)
# enqueue_op = queue.enqueue()


conv_encoder = snt.nets.ConvNet2D(
    output_channels=(16, 32, 32, 32),
    kernel_shapes=(3, 5, 5, 3),
    strides=(1, 2, 2, 2),
    paddings=(snt.SAME,))

mlp_encoder = snt.nets.MLP(output_sizes=(16,))

mlp_decoder = snt.nets.MLP(output_sizes=(1024,))
conv_decoder = snt.nets.ConvNet2DTranspose(
    output_channels=(32, 32, 16, 3),
    output_shapes=((6, 6), (12, 12), (24, 24), (48, 48)),
    kernel_shapes=(6, 5, 5, 3),
    strides=(6, 2, 2, 2),
    paddings=(snt.SAME,))

input_image = value[0]
embedding = conv_encoder(input_image)
for l in conv_encoder._layers:
    print("layer thingy: {}".format(l.input_shape[1:3]))

os = conv_encoder.transpose()._output_shapes
print("output shapes: {}".format([s() for s in os]))

embedding = snt.BatchFlatten()(embedding)
embedding = mlp_encoder(embedding)

decoded = mlp_decoder(embedding)
decoded = snt.BatchReshape(shape=(1, 1, 1024))(decoded)
decoded = conv_decoder(decoded)
reconstruction_err = tf.losses.mean_squared_error(input_image, decoded)

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(reconstruction_err)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        err, _ = sess.run([reconstruction_err, train_op])
        print("{} : reconstruction error: {}".format(i, err))
