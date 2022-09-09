import numpy as np
from .functions import softplus_forward, softplus_backward, softmax_forward, softmax_backward, convolution2d, relu_backward, relu_forward
from .optimizer import Optimizer, RMSprop, Adam

np.random.seed(0)


class Layer:

    def __init__(self):
        pass

    def fpass(self, input_data):
        pass

    def bpass(self, douts):
        pass

    def clear_grads(self):
        pass

    def update_grads(self):
        pass


class DenseLayer(Layer):

    def __init__(self, n_neurons, input_size, activation, optimizer, learning_rate):
        super().__init__()

        self.n_neurons = n_neurons
        self.input_size = input_size
        self.learning_rate = learning_rate

        if activation == 'softplus':
            self.forward_activation = softplus_forward
            self.backward_activation = softplus_backward
        elif activation == 'softmax':
            self.forward_activation = softmax_forward
            self.backward_activation = softmax_backward
        elif activation == 'relu':
            self.forward_activation = relu_forward
            self.backward_activation = relu_backward

        self.weights = np.random.rand(self.n_neurons, self.input_size) - 0.5 #* np.sqrt(2.0 / np.prod(self.input_size))
        self.dweights = np.zeros((self.n_neurons, self.input_size))
        self.biases = np.random.rand(self.n_neurons) - 0.5 #* np.sqrt(2.0 / np.prod(self.input_size))
        self.dbiases = np.zeros(self.n_neurons)
        self.Zs = np.zeros(self.n_neurons)

        self.input_data = []  # for backward pass

        if optimizer == "SGD":
            self.w_optimizer = Optimizer(self.learning_rate)
            self.b_optimizer = Optimizer(self.learning_rate)
        elif optimizer == "RMSprop":
            self.w_optimizer = RMSprop(self.dweights.shape, self.learning_rate)
            self.b_optimizer = RMSprop(self.dbiases.shape, self.learning_rate)
        elif optimizer == "Adam":
            self.w_optimizer = Adam(self.dweights.shape, self.learning_rate)
            self.b_optimizer = Adam(self.dbiases.shape, self.learning_rate)
        else:
            print(optimizer + ": Unknown method of optimising")

        self.batch_serial = 0

    def fpass(self, input_data):
        input_data = np.reshape(input_data, self.input_size)
        self.input_data = input_data
        for n in range(self.n_neurons):
            self.Zs[n] = np.dot(input_data, self.weights[n]) + self.biases[n]
        return self.forward_activation(self.Zs)

    def bpass(self, douts):
        dZs = douts * self.backward_activation(self.Zs)
        self.dbiases += dZs
        self.dweights += np.matmul(np.expand_dims(self.input_data, axis=1),
                                   np.expand_dims(dZs, axis=1).T).T
        self.batch_serial += 1
        return np.sum(dZs * self.weights.T, axis=1)

    def clear_grads(self):
        self.dweights = np.zeros((self.n_neurons, self.input_size))
        self.dbiases = np.zeros(self.n_neurons)

    def update_grads(self):
        self.weights -= self.w_optimizer.get_deltas(self.dweights / self.batch_serial)
        self.biases -= self.b_optimizer.get_deltas(self.dbiases / self.batch_serial)
        self.batch_serial = 0
        self.clear_grads()


class Conv2dLayer(Layer):

    def __init__(self, n_kernels, kernel_size, padding, input_size, activation, optimizer, learning_rate, stride=1):
        super().__init__()

        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.input_size = input_size
        self.learning_rate = learning_rate

        if activation == 'softplus':
            self.forward_activation = softplus_forward
            self.backward_activation = softplus_backward
        elif activation == 'relu':
            self.forward_activation = relu_forward
            self.backward_activation = relu_backward

        self.weights = np.random.rand(self.n_kernels, self.kernel_size, self.kernel_size) - 0.5 #* np.sqrt(2.0 / np.prod(self.input_size))
        self.dweights = np.zeros((self.n_kernels, self.kernel_size, self.kernel_size))
        self.biases = np.random.rand(1) - 0.5 #np.sqrt(2.0 / np.prod(self.input_size))
        self.dbiases = 0.0

        self.fmap_size = tuple((np.array(self.input_size[-2:]) - self.kernel_size + 2 * self.padding) + 1)
        self.Zs = np.zeros((self.n_kernels,) + self.fmap_size)

        self.input_data = []  # for backward pass

        if optimizer == "SGD":
            self.w_optimizer = Optimizer(self.learning_rate)
            self.b_optimizer = Optimizer(self.learning_rate)
        elif optimizer == "RMSprop":
            self.w_optimizer = RMSprop(self.dweights.shape, self.learning_rate)
            self.b_optimizer = RMSprop(1, self.learning_rate)
        elif optimizer == "Adam":
            self.w_optimizer = Adam(self.dweights.shape, self.learning_rate)
            self.b_optimizer = Adam(1, self.learning_rate)
        else:
            print(optimizer + ": Unknown method of optimising")

        self.batch_serial = 0

    def fpass(self, input_data):
        self.input_data = input_data
        self.Zs = np.zeros((self.n_kernels,) + self.fmap_size)
        # for 1-channel first layer
        if len(input_data.shape) > 2:
            for n in range(self.n_kernels):
                for c in range(self.input_data.shape[0]):
                    self.Zs[n] += convolution2d(input_data[c], self.weights[n], padding=self.padding,
                                                stride=self.stride)
                self.Zs[n] += self.biases
        # for multichannel hidden layers
        else:
            for n in range(self.n_kernels):
                self.Zs[n] = convolution2d(input_data, self.weights[n], padding=self.padding, stride=self.stride) \
                             + self.biases
        return self.forward_activation(self.Zs)

    def bpass(self, douts):
        if len(douts.shape) == 1:
            douts = douts.reshape((self.n_kernels,) + self.fmap_size)

        dZs = douts * self.backward_activation(self.Zs)

        if len(self.input_size) > 2:
            joined_input_data = np.sum(self.input_data, axis=0)
            for n in range(self.n_kernels):
                self.dweights[n] += convolution2d(joined_input_data, dZs[n], padding=0, stride=1)
        else:
            for n in range(self.n_kernels):
                self.dweights[n] += convolution2d(self.input_data, dZs[n], padding=0, stride=1)

        self.dbiases += np.sum(dZs)

        new_douts = np.zeros(self.input_size)
        if len(self.input_size) > 2:
            for c in range(self.input_size[0]):
                for n in range(self.n_kernels):
                    new_douts[c] += convolution2d(dZs[n],
                                                  np.flip(self.weights[n]),
                                                  padding=(self.kernel_size - 1) - self.padding, stride=1)

        self.batch_serial += 1

        return new_douts

    def clear_grads(self):
        self.dweights = np.zeros((self.n_kernels, self.kernel_size, self.kernel_size))
        self.dbiases = 0.0

    def update_grads(self):
        self.weights -= self.w_optimizer.get_deltas(self.dweights / self.batch_serial)
        self.biases -= self.b_optimizer.get_deltas(self.dbiases / self.batch_serial)
        self.batch_serial = 0
        self.clear_grads()


class MaxPooling(Layer):

    def __init__(self, input_size, filter_size, stride):
        super().__init__()
        self.input_size = input_size
        self.filter_size = filter_size
        self.stride = stride
        self.res_size = (self.input_size[0], self.input_size[1] // stride, self.input_size[2] // stride)
        self.dvalues = np.zeros(input_size)

    def fpass(self, input_data):
        self.dvalues = np.zeros(self.input_size)
        res = np.zeros(self.res_size)
        for n in range(self.input_size[0]):
            k1 = 0
            for i in range(0, self.input_size[1], self.stride):
                k2 = 0
                for j in range(0, self.input_size[2], self.stride):
                    res[n, k1, k2] = np.max(input_data[n, i:i + self.filter_size, j:j + self.filter_size])
                    amax = np.argmax(input_data[n, i:i + self.filter_size, j:j + self.filter_size])
                    assert 0 <= amax <= 3
                    self.dvalues[n, i + (amax // self.filter_size), j + (amax % self.filter_size)] += 1
                    k2 += 1
                k1 += 1
        return res

    def bpass(self, douts):
        douts = douts.reshape(self.res_size)
        for n in range(self.input_size[0]):
            k1 = 0
            for i in range(0, self.input_size[1], self.stride):
                k2 = 0
                for j in range(0, self.input_size[2], self.stride):
                    self.dvalues[n, i:i + self.filter_size, j:j + self.filter_size] = \
                        self.dvalues[n, i:i + self.filter_size, j:j + self.filter_size] * douts[n, k1, k2]
                    k2 += 1
                k1 += 1
        return self.dvalues
