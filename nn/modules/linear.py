import numpy as np

from .module import Module
from utils import sigmoid, sigmoid_deriv

class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = np.random.randn(dim_out, dim_in)
        self.bias = np.random.randn(dim_out, 1)
        self.delta = None  # gradient of feature map in this layer without activation
        self.output = None

    def forward(self, input):
        self.output = np.matmul(self.weight, input) + self.bias
        self.output = sigmoid(self.output)

        return self.output

    def backward(self, delta_top, weight_top=None, **kwargs):
        if weight_top is None:  # the topest layer
            self.delta = delta_top * sigmoid_deriv(self.output)
        else:
            self.delta = np.matmul(weight_top.T, delta_top) * sigmoid_deriv(self.output)

    def update(self, bottom_output, lr):
        if bottom_output.ndim == 4:
            bottom_output = bottom_output.reshape(bottom_output.shape[0], -1, 1)
        self.weight = self.weight - lr * np.mean(np.matmul(self.delta, bottom_output.transpose(0, 2, 1)), axis=0)
        self.bias = self.bias - lr * np.mean(self.delta, axis=0)
