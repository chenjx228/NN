import numpy as np

from .module import Module
from nn.functional import conv2d, conv2d_deriv
from utils import sigmoid, sigmoid_deriv


class Conv(Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding):
        super(Conv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = np.random.randn(dim_out, dim_in, kernel_size, kernel_size)
        self.bias = np.random.randn(dim_out, 1, 1)
        self.delta = None  # gradient of feature map in this layer without activation
        self.output = None

    def forward(self, input):
        self.output = conv2d(input, self.weight, self.padding) + self.bias
        self.output = sigmoid(self.output)

        return self.output

    def backward(self, delta_top, weight_top=None, padding_top=None, pos_top=None, **kwargs):
        if delta_top.ndim == 4:
            if pos_top is None: # top layer is Conv Module
                weight_top_trans = weight_top.transpose((1, 0, 2, 3))
                weight_top_trans = np.rot90(weight_top_trans, k=2, axes=(2, 3))
                self.delta = conv2d(delta_top, weight_top_trans, padding_top) * sigmoid_deriv(self.output)
            else:   # top layer is Pooling Module
                self.delta = np.zeros_like(self.output)
                for i in range(delta_top.shape[2]):
                    for j in range(delta_top.shape[3]):
                        pos_tmp = pos_top[:, :, i, j, :]
                        self.delta[pos_tmp[..., 0], pos_tmp[..., 1], pos_tmp[..., 2], pos_tmp[..., 3]] \
                            += delta_top[..., i, j]
                self.delta = self.delta * sigmoid_deriv(self.output)

        elif delta_top.ndim == 3:   # top layer is Linear Module
            delta = np.matmul(weight_top.T, delta_top)
            delta = delta.reshape(self.output.shape)
            self.delta = delta * sigmoid_deriv(self.output)

    def update(self, bottom_output, lr):
        self.weight = self.weight - lr * np.mean(conv2d_deriv(bottom_output, self.delta, padding=self.padding), axis=0)
        self.bias = self.bias - lr * np.mean(np.sum(self.delta, axis=(2, 3)), axis=0).reshape(self.bias.shape)
