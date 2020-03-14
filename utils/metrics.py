import numpy as np

def accuracy(predict, label):
    predict_idx = np.argmax(predict, axis=1).reshape(predict.shape[0])
    label_idx = np.argmax(label, axis=1).reshape(label.shape[0])

    acc = 100.0 * np.sum((predict_idx == label_idx)) / predict.shape[0]

    return acc