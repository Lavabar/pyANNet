from keras.datasets import mnist
from ANNet.model import Model

(train_X, train_y), (test_X, test_y) = mnist.load_data()
model = Model(train_X, train_y)
model.evaluate(test_X, test_y)
losses = model.train()
model.evaluate(test_X, test_y)
