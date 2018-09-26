
import cPickle
import idx
import numpy as np
from PIL import Image
import random


_MNIST_TRAIN_IMAGES = "/home/osushkov/Programming/vae_gan/data/train-images.idx3-ubyte"
_MNIST_TRAIN_LABELS = "/home/osushkov/Programming/vae_gan/data/train-labels.idx1-ubyte"
_MNIST_TEST_IMAGES = "/home/osushkov/Programming/vae_gan/data/t10k-images.idx3-ubyte"
_MNIST_TEST_LABELS = "/home/osushkov/Programming/vae_gan/data/t10k-labels.idx1-ubyte"
_CIFAR_TRAIN_PATH = "/home/osushkov/Programming/vae_gan/data/cifar-100-python/train"
_CIFAR_TEST_PATH = "/home/osushkov/Programming/vae_gan/data/cifar-100-python/test"


def _croppedImage(img_data):

    def max_index(data, indices, dim):
        idx = indices[0]
        for i in indices:
            idx = i
            dim_sum = np.sum(data[i, :]) if dim == 0 else np.sum(data[:, i])
            if dim_sum > 0.0:
                break
        return idx

    min_row = max_index(img_data, range(img_data.shape[0]), 0)
    max_row = max_index(img_data, list(reversed(range(img_data.shape[0]))), 0)

    min_col = max_index(img_data, range(img_data.shape[1]), 1)
    max_col = max_index(img_data, list(reversed(range(img_data.shape[1]))), 1)

    return img_data[min_row : max_row+1, min_col : max_col+1]


def _createMask(mask_size, image, image_loc):
    mask = np.zeros(mask_size, dtype=np.float32)

    y_start = image_loc[0] - image.shape[0] / 2
    y_end = y_start + image.shape[0]
    x_start = image_loc[1] - image.shape[1] / 2
    x_end = x_start + image.shape[1]

    mask[y_start : y_end, x_start : x_end] = image
    return np.expand_dims(mask, 2)


def _chooseForegroundColor(image):
    num_pixels = image.shape[0] * image.shape[1]

    avrg_r = np.sum(image[:, :, 0]) / num_pixels
    avrg_g = np.sum(image[:, :, 1]) / num_pixels
    avrg_b = np.sum(image[:, :, 2]) / num_pixels
    avrg_rgb = np.array([avrg_r, avrg_g, avrg_b], )

    candidate_colors = [
        np.array([1., 0., 0.], dtype=np.float32),  # Red
        np.array([0., 1., 0.], dtype=np.float32),  # Green
        np.array([0., 0., 1.], dtype=np.float32),  # Blue
        # np.array([0., 0., 0.], dtype=np.float32),  # Black
        # np.array([1., 1., 1.], dtype=np.float32)   # White
    ]

    best_candidate = None
    max_distance = 0.

    for col in candidate_colors:
        dist = np.linalg.norm(avrg_rgb - col)
        if best_candidate is None or dist > max_distance:
            max_distance = dist
            best_candidate = col

    assert best_candidate is not None
    return best_candidate


def _combinedImage(background, digit, overlay):
    base_width = background.shape[1]
    base_height = background.shape[1]

    digit_width = digit.shape[1]
    digit_hwidth = digit_width / 2

    digit_height = digit.shape[0]
    digit_hheight = digit_height / 2

    min_x = digit_hwidth + 1
    min_y = digit_hheight + 1

    max_x = base_width - digit_width + digit_hwidth - 1
    max_y = base_height - digit_height + digit_hheight - 1

    loc = random.randint(min_y, max_y), random.randint(min_x, max_x)
    mask = _createMask((base_height, base_width), digit, loc)
    mask = np.power(mask, 0.75)

    combined = (np.ones_like(mask, dtype=np.float32) - mask) * background + mask * overlay
    combined = np.clip(combined, 0.0, 1.0)

    norm_loc = (loc[0] / float(base_height), loc[1] / float(base_width))

    return combined, np.array(norm_loc, dtype=np.float32)


class DataProvider(object):

    def __init__(self, target_width, target_height):
        with open(_CIFAR_TEST_PATH, "rb") as fo:
            cifar = cPickle.load(fo)

            cifar = cifar["data"].reshape((-1, 3, 32, 32))
            cifar = cifar.transpose([0, 2, 3, 1])
            cifar = cifar.astype(np.float32)

        self._cifar = []
        self._overlays = []
        for  i in range(cifar.shape[0]):
            import matplotlib.pyplot as plt

            background = Image.fromarray(cifar[i], 'RGB')
            background = background.resize((target_height, target_width),
                                           Image.BILINEAR)
            background = np.asarray(background, dtype=np.float32) / 255.0

            plt.imshow(cifar[i] / 255.0)
            plt.show()
            
            self._cifar.append(background)

            overlay = np.array([[_chooseForegroundColor(background)]], dtype=np.float32)
            overlay = np.repeat(np.repeat(overlay, target_height, 0), target_width, 1)
            self._overlays.append(overlay)

        mnist = idx.readImages(_MNIST_TEST_IMAGES)

        self._mnist_images = []
        for i in range(mnist.shape[0]):
            self._mnist_images.append(_croppedImage(mnist[i]))

        self._mnist_labels = idx.readLabels(_MNIST_TEST_LABELS)

        assert len(self._cifar) == len(self._overlays)
        assert len(self._mnist_images) == len(self._mnist_labels)
        assert len(self._cifar) > 0 and len(self._mnist_images) > 0


    def __call__(self):
        ci = random.randrange(len(self._cifar))
        mi = random.randrange(len(self._mnist_images))

        img, loc = _combinedImage(self._cifar[ci],
                                  self._mnist_images[mi],
                                  self._overlays[ci])

        return img, loc, self._mnist_labels[mi]


