import numpy as np
import layer
import relu


# Function that calculates the cross-entropy loss of the model
# param u: Weighted inputs of the final layer
# param y: Labels of the training set
# return: Cross-entropy loss of the model
def cross_entropy_softmax(u, y):
    correct = u[np.arange(len(u)), y]
    loss = - correct + np.log(np.sum(np.exp(u), axis=-1))
    return loss


# Function that calculates the gradient of the cross-entropy loss of the model
# param u: Weighted inputs of the final layer
# param y: Labels of the training set
# return: Gradient of the cross-entropy loss of the model
def cross_entropy_softmax_grad(u, y):
    onehot_y = np.zeros_like(u)
    onehot_y[np.arange(len(u)), y] = 1
    softmax = np.exp(u) / np.exp(u).sum(axis=-1, keepdims=True)
    return (softmax - onehot_y) / u.shape[0]


# Class that implements a Multilayer Perceptron
class MLP:

    # Constructor
    # param input_layer_neurons: Number of neurons for the input layer
    # param n_layers: Number of layers
    # param: n_neurons: List that contains the number of neurons for each layer except for the input layer.
    # param learning_rate: Learning rate
    # param n_classes: Number of classes of the dataset
    def __init__(self, input_layer_neurons, n_layers, n_neurons, learning_rate, n_classes):
        self.network = [layer.Layer(input_layer_neurons, n_neurons[0], learning_rate),
                        relu.ReLU()]
        j = 0
        for i in range(1, n_layers):
            if i != n_layers - 1:
                self.network.append(layer.Layer(n_neurons[i - 1], n_neurons[i], learning_rate))
                self.network.append(relu.ReLU())
            else:
                self.network.append(layer.Layer(n_neurons[i - 1], n_classes, learning_rate))
            j += 1

    # Function that implements the feed forward algorithm
    # param x: Input samples
    # return: List that contains the weighted inputs (u) and the outputs of the activation functions (y) of each layer
    def feed_forward(self, x):
        activations = []
        # Looping through each layer
        for l in self.network:
            activations.append(l.forward(x))
            # Updating the next layer's input to the last layer output's
            x = activations[-1]
        return activations

    # Function that implements the training phase of the MLP
    # param x: Training samples
    # param y: Labels of the training samples
    def train(self, x, y):
        # Getting the list that contains the weighted inputs (u) and the outputs of the activation functions (y) of each
        # layer with the feed-forward algorithm
        layer_activations = self.feed_forward(x)

        # Adding the original inputs (x) at the beginning of the list
        layer_inputs = [x] + layer_activations
        u_output = layer_activations[-1]

        # Calculating the gradient of the cross-entropy loss
        loss_grad = cross_entropy_softmax_grad(u_output, y)

        # Back-propagation
        for layer_index in reversed(range(len(self.network))):
            loss_grad = self.network[layer_index].backward(layer_inputs[layer_index], loss_grad)

    # Function that evaluates the model on a test/validation set
    # param x: Test/validation samples
    # param y: Labels of the test/validation samples
    # return: SoftMax output, loss
    def evaluate(self, x, y):
        u = self.feed_forward(x)[-1]
        loss = cross_entropy_softmax(u, y)
        softmax = np.exp(u) / np.exp(u).sum(axis=-1, keepdims=True)
        return softmax, loss

    # Function that gets the current network
    # return: Current network
    def get_network(self):
        return self.network
