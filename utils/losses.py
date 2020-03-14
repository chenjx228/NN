import numpy as np

def L2Loss(predict, label):
    return np.sum((predict - label) ** 2) / predict.shape[0]

def L2Loss_deriv(predict, label):
    return 2 * (predict - label)