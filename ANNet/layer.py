import numpy as np
from .functions import softplus_forward, softplus_backward, softmax_forward, softmax_backward
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

        self.weights = np.random.rand(self.n_neurons, self.input_size) - 0.5
        self.dweights = np.zeros((self.n_neurons, self.input_size))
        self.biases = np.zeros(self.n_neurons)
        self.dbiases = np.zeros(self.n_neurons)
        self.Zs = np.zeros(self.n_neurons)

        self.input_data = []  # for backward pass

        if optimizer == "SGD":
            self.w_optimizer = Optimizer(self.learning_rate)
            self.b_optimizer = Optimizer(self.learning_rate)
        elif optimizer == "RMSprop":
            self.w_optimizer = RMSprop((self.n_neurons, self.input_size), self.learning_rate)
            self.b_optimizer = RMSprop((self.n_neurons,), self.learning_rate)
        elif optimizer == "Adam":
            self.w_optimizer = Adam((self.n_neurons, self.input_size), self.learning_rate)
            self.b_optimizer = Adam((self.n_neurons,), self.learning_rate)
        else:
            print(optimizer + ": Unknown method of optimising")

        self.batch_serial = 0

    def fpass(self, input_data):
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

    def __init__(self):
        super().__init__()
