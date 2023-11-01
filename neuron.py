
import numpy as np
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.2, 0.8, -0.5],
           [0.5, -0.91, 0.26],
           [-0.26, -0.27, 0.17]]

biases = [2, 3, 0.5]
biases2 = [2, 3, 0.5]

output1 = np.dot(inputs, np.array(weights).T) + biases

output2 = np.dot(output1, np.array(weights2).T) + biases2
print(output1,"\n",output2)