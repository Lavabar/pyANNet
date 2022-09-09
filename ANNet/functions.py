import numpy as np


def softplus_forward(x):
    return np.log(1 + np.exp(x))


# good softmax?
#   np.log(np.sum(np.exp(x)))
def softmax_forward(x):
    expos = np.exp(x - np.max(x))
    return expos / expos.sum(axis=0, keepdims=True)


def softmax_backward(x):
    return np.ones(x.shape[0])


def softplus_backward(x):
    return 1 / (1 + np.exp(-1 * x))


def cross_entropy_loss(output, target):
    return (-1) * np.dot(np.log(output), target)
