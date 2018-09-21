
import data_provider
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf


def _showRandom(data_source):
    img, loc, label = data_source()

    print("loc: {}".format(loc))
    print("label: {}".format(label))

    plt.imshow(img, cmap="gray")
    plt.show()


data_source = data_provider.DataProvider(48, 48)

# for _ in range(100):
#     _showRandom(data_source)

def _generator():
    yield data_source()

dataset = tf.data.Dataset.from_generator(
    generator=_generator, output_types=(tf.float32, tf.float32, tf.float32))

value = dataset.make_one_shot_iterator().get_next()
# sample_op = tf.py_func(data_source, [], [tf.float32, tf.float32, tf.float32])
# queue = tf.FIFOQueue(capacity=128, dtype=tf.float32)
# enqueue_op = queue.enqueue()

with tf.Session() as sess:
    print(sess.run(value))
