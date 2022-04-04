import numpy as np


# Class that implements the ReLU activation function
class ReLU:

    # Constructor
    def __init__(self):
        pass

    # Function that passes the weighted inputs through the ReLU activation function
    # param input: Input sample
    # return: Output of the ReLU activation function
    def forward(self, input):
        return np.maximum(0, input)

    # Function that passes the inputs of the next layer through the gradient of ReLU and calculates the gradient of the
    # loss for that layer
    # param input: Input sample
    # param grad_output: Gradient of the loss of the next layer
    # return: Gradient of the ReLU activation function
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad
