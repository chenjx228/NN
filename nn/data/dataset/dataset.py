import numpy as np

from .transforms import *

class Dataset(object):
    def __init__(self, train, test, mode='Train'):
        super(Dataset, self).__init__()
        self.train = train
        self.test = test
        self.mode = mode

        if self.mode == 'Train':
            self.data = self.train
        elif self.mode == 'Test':
            self.data = self.test

    def return_data(self):
        return self.data

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = to_input_array(img)

        return img, label

    def __len__(self):
        return len(self.data)
