import numpy as np
import math
"""
This script contains an activation functions - non part of the ANN or main
"""
np.random.seed(0)  # seed for random number generator
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]  # input data to neural network

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]  # input data to neural network
output = []


def step_activation(activate_input):
    for i in activate_input:
        if i > 0:
            output.append(1)
        elif i <= 0:
            output.append(0)
    return output


def ReLU_activation(activate_input):
    for i in activate_input:
        if i >= 0:
            output.append(i)
        elif i <= 0:
            output.append(0)
    return output


def Sigmoid_activation(activate_input):
    for i in activate_input:
        output.append(1 / (1 + np.exp(-i)))
    return output


softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
print(loss)
