import numpy as np
import copy


class Optimizer:
    def __init__(self, grads_structure, learning_rate):
        self.learning_rate = learning_rate
        self.epsilon = 1.0e-4

        self.running_Eg2 = grads_structure
        self.gamma = 0.7

        self.beta1 = 0.3
        self.beta2 = 0.7
        self.running_ms = copy.deepcopy(grads_structure)
        self.running_vs = copy.deepcopy(grads_structure)

    def sgd_optimizer(self, gt):
        return self.learning_rate * gt

    def update_Eg2(self, gt, k):
        self.running_Eg2[k] = self.gamma * self.running_Eg2[k] + (1.0 - self.gamma) * np.square(gt)

    def update_ms(self, gt, k):
        self.running_ms[k] = self.beta1 * self.running_ms[k] + (1.0 - self.beta1) * gt

    def update_vs(self, gt, k):
        self.running_vs[k] = self.beta2 * self.running_vs[k] + (1.0 - self.beta2) * np.square(gt)

    def rmsprop_optimizer(self, gt, k):
        self.update_Eg2(gt, k)
        return self.learning_rate * gt / np.sqrt(self.running_Eg2[k] + self.epsilon)

    def adam_optimizer(self, gt, k, t):
        self.update_ms(gt, k)
        self.update_vs(gt, k)
        m_hat = self.running_ms[k] / (1.0 - np.power(self.beta1, t))
        v_hat = self.running_vs[k] / (1.0 - np.power(self.beta2, t))
        return self.learning_rate * m_hat / np.sqrt(v_hat + self.epsilon)
