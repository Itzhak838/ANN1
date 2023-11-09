import numpy as np
import matplotlib.pyplot as plt
import ANN


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


def main():
    X, y = spiral_data(samples=100, classes=3)
    # plt.scatter(X[:, 0], X[:, 0], c=y, s=40)
    # plt.show()

    dense1 = ANN.Layer_Dense(2, 3)
    activation1 = ANN.Activation_Softmax()

    dense2 = ANN.Layer_Dense(3, 5)
    activation2 = ANN.Activation_Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])

    loss_function = ANN.Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    print("Loss:", loss)


if __name__ == "__main__":
    main()


