import numpy as np


# Class that implements a layer of the Multilayer Perceptron network
class Layer:

    # Constructor
    # param input_units: Number of neurons
    # param output_units: Number of neurons of the next layer
    # learning_rate: Learning rate
    def __init__(self, input_units, output_units, learning_rate):
        self.learning_rate = learning_rate
        # Initializing weights from the normal distribution for mean=0 and std=np.sqrt(2 / (input_units + output_units))
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    # Function that calculates the weighted inputs of each layer (u = x.T * w + b)
    # param input: Input samples of this layer
    # return: Weighted inputs (u)
    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    # Function that updates the weights and biases of this layer during back-propagation
    # param input: Input samples of this layer
    # grad_output: Gradient of the loss of the next layer
    # return: Gradient of the loss for the previous layer
    def backward(self, input, grad_output):
        # Calculating the gradient of the loss that will be propagated back to the previous layers
        grad_input = np.dot(grad_output, self.weights.T)

        # Calculating Δw and Δb
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        # Updating weights and biases
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

    # Function that gets the weights of the layer
    # return: Weights of the layer
    def get_weights(self):
        return self.weights

    # Function that sets the weights of the layer
    # param w: Weights that will be loaded into the layer
    def set_weights(self, w):
        self.weights = w

    # Function that gets the biases of the layer
    # return: Biases of the layer
    def get_biases(self):
        return self.biases

    # Function that sets the biases of the layer
    # param b: Biases that will be loaded into the layer
    def set_biases(self, b):
        self.biases = b
