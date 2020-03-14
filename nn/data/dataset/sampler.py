import numpy as np


class Sampler(object):
    def __init__(self, base_sampler, shuffle, batch_size, drop_last):
        super(Sampler, self).__init__()

        self.sampler = base_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        if shuffle:
            np.random.shuffle(self.sampler)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

