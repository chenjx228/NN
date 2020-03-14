import numpy as np


def parse_data(data, class_num):
    imgs, labels = data
    imgs = np.stack(imgs)
    labels = np.array(labels, np.int32)

    labels_onehot = np.zeros((labels.shape[0], class_num), np.int32)
    labels_onehot[range(labels.shape[0]), labels] = 1
    labels_onehot = labels_onehot[..., np.newaxis]

    return imgs, labels_onehot
