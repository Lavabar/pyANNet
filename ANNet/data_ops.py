import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class DataLoader:

    def __init__(self, x_data, y_data, minibatch_size):
        self.minibatch_size = minibatch_size

        self.X = np.reshape(x_data, (x_data.shape[0], -1)) / x_data.max()
        self.Y = to_categorical(y_data)

        self.dataset_size = x_data.shape[0]

    def get_batch(self):
        self.X, self.Y = unison_shuffled_copies(self.X, self.Y)
        for i in range(0, len(self.X) - self.minibatch_size - 1, self.minibatch_size):
            yield self.X[i:i + self.minibatch_size], self.Y[i:i + self.minibatch_size]