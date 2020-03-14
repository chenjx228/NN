import numpy as np


def to_input_array(img):
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        img = img.transpose(0, 3, 1, 2)

    return img
