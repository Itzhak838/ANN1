"""
this is a classes script-based implementation of
a simple artificial neural network (ANN)
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)  # seed for random number generator

def spiral_data(samples, classes):
    """
    create a spiral dataset to test the ANN
    """
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# initialize weights - random values -1 to 1 ranged to avoid nuber explosion
class Layer_Dense:
    """
    this is a class that represents a layer of neurons
    and calculate the output of the layer by multiplying the
     inputs by the weights and adding the biases.
    """
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    """
    this is a class that represents the ReLU
    activation function.
    the output is 0 if the input is negative
    and the input itself if it is positive.
    """
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    """
    this is a class that represents the Softmax
     activation function.
    the output is the probability of each class
     after normalization and exponentiation.
    """
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    """
    this is a class that represents the loss
    function (metric for error) of the ANN
    """
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods












X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 0], c=y, s=40, camp='brg')
plt.show()

dense1 = Layer_Dense(2, 3)
activation1 = Activation_Softmax()

dense2: Layer_Dense = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
