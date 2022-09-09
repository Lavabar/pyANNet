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


def convolution2d(x, kernel, padding, stride):
    kernel_size = kernel.shape[0]
    res_size = tuple((np.array(x.shape) - kernel_size + 2 * padding) + 1)
    new_x = np.pad(x, padding, constant_values=0)
    res = np.zeros(res_size)
    kernel = kernel.flatten()
    for i in range(0, res_size[0], stride):
        for j in range(0, res_size[1], stride):
            res[i, j] = np.dot(new_x[i:i+kernel_size, j:j+kernel_size].flatten(), kernel)
    return res


if __name__ == '__main__':
    a = np.arange(64).reshape((8, 8))
    k = np.arange(9).reshape((3, 3))
    res = convolution2d(a, k, 1, 1)
    print(res.shape)
    print(res)
