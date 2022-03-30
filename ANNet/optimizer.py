import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get_deltas(self, gt):
        return self.learning_rate * gt


class RMSprop(Optimizer):

    def __init__(self, grads_structure, learning_rate):
        super().__init__(learning_rate)
        self.epsilon = 1e-07

        self.running_Eg2 = np.zeros(grads_structure)
        self.gamma = 0.7

    def update_Eg2(self, gt):
        self.running_Eg2 = self.gamma * self.running_Eg2 + (1.0 - self.gamma) * np.square(gt)

    def get_deltas(self, gt):
        self.update_Eg2(gt)
        return self.learning_rate * gt / np.sqrt(self.running_Eg2 + self.epsilon)


class Adam(Optimizer):

    def __init__(self, grads_structure, learning_rate):
        super().__init__(learning_rate)
        self.epsilon = 1e-07
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.running_ms = np.zeros(grads_structure)
        self.running_vs = np.zeros(grads_structure)

        self.t = 0

    def update_ms(self, gt):
        self.running_ms = self.beta1 * self.running_ms + (1.0 - self.beta1) * gt

    def update_vs(self, gt):
        self.running_vs = self.beta2 * self.running_vs + (1.0 - self.beta2) * np.square(gt)

    def get_deltas(self, gt):
        self.t += 1
        self.update_ms(gt)
        self.update_vs(gt)
        m_hat = self.running_ms / (1.0 - np.power(self.beta1, self.t))
        v_hat = self.running_vs / (1.0 - np.power(self.beta2, self.t))
        return self.learning_rate * m_hat / np.sqrt(v_hat + self.epsilon)
