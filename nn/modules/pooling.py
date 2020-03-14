import numpy as np

from .module import Module
from nn.functional import conv2d, maxpool

from utils.activate_func import sigmoid_deriv

class Pooling(Module):
    def __init__(self, kernel_size, padding, stride):
        super(Pooling, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.delta = None  # gradient of feature map in this layer without activation
        self.pos = None  # related position of pooling operation
        self.output = None

    def forward(self, input):
        NotImplementedError

    def backward(self, delta_top, weight_top, padding_top=None, **kwargs):
        NotImplementedError

    def update(self, bottom_output, lr):
        NotImplementedError


class MaxPool(Pooling):
    def __init__(self, kernel_size, padding, stride):
        super(MaxPool, self).__init__(kernel_size, padding, stride)

    def forward(self, input):
        self.output, self.pos = maxpool(input, self.kernel_size, self.padding, self.stride)

        return self.output

    def backward(self, delta_top, weight_top=None, padding_top=None, pos_top=None, **kwargs):
        if delta_top.ndim == 4:
            if pos_top is None: # top layer is Conv Module
                weight_top_trans = weight_top.transpose((1, 0, 2, 3))
                weight_top_trans = np.rot90(weight_top_trans, k=2, axes=(2, 3))
                self.delta = conv2d(delta_top, weight_top_trans, padding_top)
            else:   # top layer is Pooling Module
                self.delta = np.zeros_like(self.output)
                for i in range(delta_top.shape[3]):
                    for j in range(delta_top.shape[4]):
                        pos_tmp = pos_top[:, :, i, j, :]
                        self.delta[pos_tmp[..., 0], pos_tmp[..., 1], pos_tmp[..., 2], pos_tmp[..., 3]] \
                            += delta_top[..., i, j]

        elif delta_top.ndim == 3:   # top layer is Linear Module
            delta = np.matmul(weight_top.T, delta_top)
            delta = delta.reshape(self.output.shape)
            self.delta = delta

    def update(self, *args, **kwargs):
        pass
