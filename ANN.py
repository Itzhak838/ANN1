"""
this is a classes script-based implementation of
a simple Artificial Neural Network (ANN)
"""
import numpy as np

np.random.seed(0)  # seed for random number generator


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
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


def main():
    print("This main function is not reached if the file is imported")


if __name__ == "__main__":
    main()
