import copy
import numpy as np

from tqdm import trange, tqdm
from .functions import cross_entropy_loss, softplus_forward, softplus_backward, softmax_forward
from .optimizer import Optimizer
from .data_ops import to_categorical, unison_shuffled_copies

np.random.seed(0)


class Model:

    def __init__(self, x_data, y_data):

        self.input_size = 28 * 28

        self.X = np.reshape(x_data, (x_data.shape[0], self.input_size)) / 255
        self.Y = to_categorical(y_data)

        self.hidden_layers = 2
        self.output_size = 10
        self.n_neurons = (self.input_size, 128, 64, self.output_size)

        self.minibatch_size = 128
        self.epochs = 3
        self.learning_rate = 0.01

        self.weights = []
        for i in range(self.hidden_layers + 1):
            self.weights.append(np.random.rand(self.n_neurons[i + 1], self.n_neurons[i]) - 0.5)

        self.biases = [np.zeros(self.n_neurons[i + 1]) for i in range(self.hidden_layers + 1)]
        self.dbiases = [np.zeros(self.n_neurons[i + 1]) for i in range(self.hidden_layers + 1)]

        self.dweights = []
        for i in range(self.hidden_layers + 1):
            self.dweights.append(np.zeros((self.n_neurons[i + 1], self.n_neurons[i])))

        self.io_values = []
        self.cost_function = cross_entropy_loss
        self.hidden_activation = softplus_forward
        self.hidden_back_activation = softplus_backward
        self.output_activation = softmax_forward
        self.W_optimizer = Optimizer(copy.deepcopy(self.dweights), self.learning_rate)
        self.b_optimizer = Optimizer(copy.deepcopy(self.dbiases), self.learning_rate)

    def get_batch(self):
        self.X, self.Y = unison_shuffled_copies(self.X, self.Y)
        for i in range(0, len(self.X) - self.minibatch_size - 1, self.minibatch_size):
            yield self.X[i:i + self.minibatch_size], self.Y[i:i + self.minibatch_size]

    def fpass(self, input_data):
        self.io_values = [np.zeros(n) for n in self.n_neurons]
        self.io_values[0] += input_data
        for i in range(1, self.hidden_layers + 2):
            if i - 1 != 0:
                for n in range(self.n_neurons[i]):
                    self.io_values[i][n] = np.dot(self.hidden_activation(self.io_values[i - 1]),
                                                  self.weights[i - 1][n]) + self.biases[i - 1][n]
            else:
                for n in range(self.n_neurons[i]):
                    self.io_values[i][n] = np.dot(self.io_values[i - 1], self.weights[i - 1][n]) + self.biases[i - 1][n]
            if i == self.hidden_layers + 2 - 1:
                self.io_values[i] = self.output_activation(self.io_values[i])

    def bpass(self, target):
        dz_values = [np.zeros(n) for n in self.n_neurons[1:]]
        for i in range(self.hidden_layers, -1, -1):
            j = i + 1
            if i == self.hidden_layers:
                dz_values[i] = self.io_values[j] - target
            else:
                dz_values[i] = np.sum(dz_values[i + 1] * self.weights[i + 1].T, axis=1) * self.hidden_back_activation(
                    self.io_values[j])
            if i != 0:
                self.dweights[i] += np.matmul(np.expand_dims(self.hidden_activation(self.io_values[j - 1]), axis=1),
                                              np.expand_dims(dz_values[i], axis=1).T).T
            else:
                self.dweights[i] += np.matmul(np.expand_dims(self.io_values[j - 1], axis=1),
                                              np.expand_dims(dz_values[i], axis=1).T).T
        for i in range(len(self.dbiases)):
            self.dbiases[i] += dz_values[i]

    def train(self):
        t = 1
        plot_losses = []
        for i in range(self.epochs):
            batch_generator = self.get_batch()
            progress_bar = tqdm(batch_generator, total=self.X.shape[0] // self.minibatch_size)
            for x_batch, y_batch in progress_bar:
                losses = 0.0
                for j in range(len(x_batch)):
                    self.fpass(x_batch[j])
                    loss = self.cost_function(self.io_values[-1], y_batch[j])
                    self.bpass(y_batch[j])
                    losses += loss
                    plot_losses.append(loss)
                progress_bar.set_description(f"Epoch: {i}, CE av loss: {losses / self.minibatch_size}")
                for k in range(len(self.weights)):
                    self.weights[k] -= self.W_optimizer.adam_optimizer(self.dweights[k] / self.minibatch_size, k, t)
                for k in range(len(self.biases)):
                    self.biases[k] -= self.b_optimizer.adam_optimizer(self.dbiases[k] / self.minibatch_size, k, t)
                self.dweights = []
                for k in range(self.hidden_layers + 1):
                    self.dweights.append(np.zeros((self.n_neurons[k + 1], self.n_neurons[k])))
                self.dbiases = [np.zeros(self.n_neurons[i + 1]) for i in range(self.hidden_layers + 1)]
            t += 1
        return plot_losses

    def evaluate(self, x_test, y_test):
        x_test = np.reshape(x_test, (x_test.shape[0], self.input_size)) / 255
        res = np.zeros(x_test.shape[0])
        for i in trange(len(x_test)):
            self.fpass(x_test[i])
            res[i] += np.argmax(self.io_values[-1]) == y_test[i]
        print(f"\nEvaluation score (accuracy): {res.sum() / res.shape[0]}")
