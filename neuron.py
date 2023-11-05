"""
This is a simple neuron class to initialize and calculate a neuron a layer
"""
import numpy as np

np.random.seed(0)  # seed for random number generator
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]  # input data to neural network


# initialize weights - random values -1 to 1 ranged to avoid nuber explosion
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)
layer1.forward(X)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)
print(activation1.output)
