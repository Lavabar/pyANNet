import numpy as np

from tqdm import trange, tqdm
from .functions import cross_entropy_loss
from .data_ops import DataLoader
from .layer import DenseLayer, Conv2dLayer, MaxPooling


class Model:

    def __init__(self, x_data, y_data):

        self.input_size = (28, 28)
        self.output_size = 10

        self.minibatch_size = 128
        self.epochs = 3
        self.learning_rate = 0.01

        self.layers = [
            Conv2dLayer(n_kernels=6, kernel_size=3, padding=(3 // 2), input_size=self.input_size, activation="softplus",
                        optimizer="Adam", learning_rate=self.learning_rate),
            Conv2dLayer(n_kernels=6, kernel_size=3, padding=(3 // 2), input_size=(6, 28, 28), activation="softplus",
                        optimizer="Adam", learning_rate=self.learning_rate),
            MaxPooling(input_size=(6, 28, 28), filter_size=2, stride=2),
            Conv2dLayer(n_kernels=16, kernel_size=3, padding=0, input_size=(6, 14, 14), activation="softplus",
                        optimizer="Adam", learning_rate=self.learning_rate),
            Conv2dLayer(n_kernels=16, kernel_size=3, padding=0, input_size=(16, 12, 12), activation="softplus",
                        optimizer="Adam", learning_rate=self.learning_rate),
            MaxPooling(input_size=(16, 10, 10), filter_size=2, stride=2),
            DenseLayer(n_neurons=120, input_size=400, activation="softplus", optimizer="Adam",
                       learning_rate=self.learning_rate),
            DenseLayer(n_neurons=84, input_size=120, activation="softplus", optimizer="Adam",
                       learning_rate=self.learning_rate),
            DenseLayer(n_neurons=self.output_size, input_size=84, activation="softmax", optimizer="Adam",
                       learning_rate=self.learning_rate)
        ]
        self.data_loader = DataLoader(x_data, y_data, self.minibatch_size)
        self.cost_function = cross_entropy_loss

    def fpass(self, input_data):

        io = input_data
        for layer in self.layers:
            io = layer.fpass(io)
        return io

    def bpass(self, douts):

        io = douts
        for layer in self.layers[::-1]:
            io = layer.bpass(io)

    def train(self):
        plot_losses = []
        for i in range(self.epochs):
            batch_generator = self.data_loader.get_batch()
            progress_bar = tqdm(batch_generator, total=self.data_loader.dataset_size // self.minibatch_size)
            for x_batch, y_batch in progress_bar:
                losses = 0.0
                for j in range(len(x_batch)):
                    out = self.fpass(x_batch[j])
                    loss = self.cost_function(out, y_batch[j])
                    self.bpass(out - y_batch[j])
                    losses += loss
                    plot_losses.append(loss)
                progress_bar.set_description(f"Epoch: {i}, av loss: {losses / self.minibatch_size}")
                for layer in self.layers:
                    layer.update_grads()
        return plot_losses

    def evaluate(self, x_test, y_test):
        #x_test = np.reshape(x_test, (x_test.shape[0], self.input_size)) / 255
        x_test = x_test / 255
        res = np.zeros(x_test.shape[0])
        for i in trange(len(x_test)):
            res[i] += np.argmax(self.fpass(x_test[i])) == y_test[i]
        print(f"\nEvaluation score (accuracy): {res.sum() / res.shape[0]}")
