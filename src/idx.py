import struct
import numpy as np


def _readIdxInt(file):
    data = file.read(4)
    return struct.unpack(">i", data)[0]


def readImages(path):
    with open(path, "rb") as f:
        mn = _readIdxInt(f)
        if mn != 2051:
            print("File did not contain expected magic number")
            return None

        num_entries = _readIdxInt(f)
        img_width = _readIdxInt(f)
        img_height = _readIdxInt(f)

        if img_width <= 0 or img_height <= 0 or num_entries <= 0:
            return None

        result = np.empty((num_entries, img_height, img_width), dtype=np.float32)

        for e in range(num_entries):
            for y in range(img_height):
                for x in range(img_width):
                    pixel = f.read(1)
                    result[e, y, x] = struct.unpack("B", pixel)[0] / 255.0

        return result


def readLabels(path):
    with open(path, "rb") as f:
        mn = _readIdxInt(f)
        if mn != 2049:
            print("File did not contain expected magic number")
            return None

        num_entries = _readIdxInt(f)

        result = np.zeros((num_entries, 10), dtype=np.float32)

        for e in range(num_entries):
            digit_buf = f.read(1)
            digit = struct.unpack("B", digit_buf)[0]
            assert 0 <= digit <= 9

            result[e, digit] = 1.0

        return result
