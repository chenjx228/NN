# from __future__ import absolute_import

import os.path as osp
import struct
import numpy as np
import matplotlib.pyplot as plt

from nn.data.dataset import Dataset

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    Decode IDX3 File

    Reference URL:
        - https://blog.csdn.net/jiede1/article/details/77099326

    :param idx3_ubyte_file: idx3 file path
    :return: imgs in the form of numpy.array (N , H, W)
    """
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images.reshape(num_images, num_rows, num_cols)

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    Decode IDX1 File

    Reference URL:
        - https://blog.csdn.net/jiede1/article/details/77099326

    :param idx3_ubyte_file: idx1 file path
    :return: labels in the form of numpy.array (N ,)
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


class MNIST(Dataset):
    dataset_dir = 'MNIST'

    def __init__(self, root,  **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_imgs_path = osp.join(self.dataset_dir, 'train-images.idx3-ubyte')
        self.train_labels_path = osp.join(self.dataset_dir, 'train-labels.idx1-ubyte')
        self.test_imgs_path = osp.join(self.dataset_dir, 't10k-images.idx3-ubyte')
        self.test_labels_path = osp.join(self.dataset_dir, 't10k-labels.idx1-ubyte')

        train = self._process_dir(self.train_imgs_path, self.train_labels_path)
        test = self._process_dir(self.test_imgs_path, self.test_labels_path)
        
        super(MNIST, self).__init__(train=train, test=test, **kwargs)

    @staticmethod
    def _process_dir(img_path, label_path):
        imgs = decode_idx3_ubyte(img_path)
        labels = decode_idx1_ubyte(label_path)

        data = list()
        for i in range(imgs.shape[0]):
            # img = imgs[i] * 1.0 / 255
            img = imgs[i]
            label = labels[i]
            data.append((img, label))

        return data
