import numpy as np

from nn.data.dataset.sampler import Sampler


class DataLoader(object):
    def __init__(self, dataset, base_sampler=None, shuffle=False, batch_size=1):
        super(DataLoader, self).__init__()
        self.dataset = dataset
        if base_sampler is None:
            base_sampler = np.arange(len(self.dataset))
        self.sampler = Sampler(base_sampler, shuffle, batch_size, drop_last=False)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.sampler)


class DataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._index_sampler = loader.sampler
        self._sampler_iter = iter(self._index_sampler)

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)

    def __next__(self):
        index = self._next_index()
        data = [self._dataset[idx] for idx in index]
        data = np.array(data)

        return data.T

    def __len__(self):
        return len(self._index_sampler)




