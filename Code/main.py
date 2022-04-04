import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
import time
import mlp
from loadDataset import load_dataset
from plotPredictions import show_most_incorrect_predictions, show_most_correct_predictions

# Specifying a random seed in order to have the same random environment for every run of the program
np.random.seed(42)

# Loading the training and test sets
x_train, y_train, x_test, y_test = load_dataset()

# Initializing MLP hyperparameters
learning_rate = 0.1
n_layers = 5
n_neurons = [500, 250, 150, 100]
n_classes = 10
batch_size = 32
epochs = 30

# Initializing Multilayer Perceptron
network = mlp.MLP(x_train.shape[1], n_layers, n_neurons, learning_rate, n_classes)

# Initializing the minimum validation loss and maximum validation accuracy
min_val_loss = float('inf')
max_val_acc = 0

# Initializing optimal weights and biases
w = []
b = []

# Initializing the split percentage of the training set into a training and a validation set
split = 0.9
n_train_examples = int(len(x_train) * split)

# Splitting the whole training set into a smaller training set and a validation set
x_train_split = x_train[:n_train_examples]
y_train_split = y_train[:n_train_examples]
x_val = x_train[n_train_examples:]
y_val = y_train[n_train_examples:]

total_time = time.time()

# Training and evaluating the Multilayer Perceptron classifier on the encoded dataset
clear_output()
print("MLP TRAINING...")
clear_output()
for epoch in range(epochs):
    epoch_time = time.time()
    # Batch training
    n_iterations = len(y_train_split) - batch_size + 1
    for i in tqdm(range(0, n_iterations, batch_size), desc="Training Epoch " + str(epoch + 1), leave=False):
        network.train(x_train_split[i:i + batch_size], y_train_split[i:i + batch_size])
    epoch_time = time.time() - epoch_time
    # Calculating the loss and accuracy for the split training set for each epoch
    train_softmax, train_loss_array = network.evaluate(x_train_split, y_train_split)
    train_pred = train_softmax.argmax(axis=-1)
    train_loss = np.mean(train_loss_array)
    train_acc = np.mean(train_pred == y_train_split)

    # Calculating the loss and accuracy for the validation set for each epoch
    val_softmax, val_loss_array = network.evaluate(x_val, y_val)
    val_pred = val_softmax.argmax(axis=-1)
    val_acc = np.mean(val_pred == y_val)
    val_loss = np.mean(val_loss_array)

    clear_output()
    print(f"Epoch {epoch + 1} / {epochs} | Epoch Time: {round(epoch_time * 1000)} ms")
    print(f"Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Valid loss: {val_loss:.3f} | Valid Accuracy: {val_acc * 100:.2f}%")

    # If the model achieved a validation accuracy higher than the previous highest one, the optimal weights and biases
    # are updated. If the validation accuracy is equal to the previous highest one, the optimal weights and biases are
    # updated if the validation loss is lower that the previous minimum loss.
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        w.clear()
        b.clear()
        for i in range(len(network.get_network()), 2):
            w.append(network.get_network()[i].get_weights())
            b.append(network.get_network()[i].get_biases())
    elif val_acc == max_val_acc:
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            w.clear()
            b.clear()
            for i in range(len(network.get_network()), 2):
                w.append(network.get_network()[i].get_weights())
                b.append(network.get_network()[i].get_biases())
total_time = time.time() - total_time
print(f"Total Training Time: {round(total_time * 1000)} ms")

# Loading the weights and biases that achieved the best validation accuracy in order to evaluate the model on the whole
# training set and on the test set.
j = 0
for i in range(len(network.get_network()), 2):
    network.get_network()[i].set_weights(w[j])
    network.get_network()[i].set_biases(b[j])
    j += 1

# Evaluating the model that achieved the best validation accuracy on the whole training set
train_softmax, train_loss = network.evaluate(x_train, y_train)
train_pred = train_softmax.argmax(axis=-1)
train_loss = np.mean(train_loss)
train_acc = np.mean(train_pred == y_train)

# Displaying the model's performance of the whole training set
print(f"Whole Training Set Loss: {train_loss:.3f} | Whole Training Set Accuracy: {train_acc * 100:.2f}%")

# Evaluating the model that achieved the best validation accuracy on the test set
test_softmax, test_loss = network.evaluate(x_test, y_test)
test_pred = test_softmax.argmax(axis=-1)
test_loss = np.mean(test_loss)
test_acc = np.mean(test_pred == y_test)

# Displaying the model's performance of the test set
print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc * 100:.2f}%")

# Showing images and information about the most correct predictions of the model, one for each digit
show_most_correct_predictions(x_test, y_test, test_softmax)

# Showing images and information about the most incorrect predictions of the model, one for each digit
show_most_incorrect_predictions(x_test, y_test, test_softmax)
