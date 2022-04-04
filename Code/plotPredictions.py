import matplotlib.pyplot as plt
import numpy as np


# Function that plots images for the most correct predictions, one for each digit
# param x: Test samples
# param y: Labels of the test samples
# param probabilities: Probabilities for every prediction that the network made for the labels of the test samples
def show_most_correct_predictions(x, y, probabilities):
    # Array that holds the probabilities and predictions for the most correct predictions, one for each digit
    correct_predictions = np.zeros((10, 3))

    # Array that holds the images whose predicted labels are among the most correct predictions.
    correct_prediction_images = np.zeros((10, 784))

    # Finding the most correct predictions, one for each digit
    for i in range(len(x)):
        if probabilities[i].argmax(axis=-1) == y[i]:
            if probabilities[i][probabilities[i].argmax(axis=-1)] > correct_predictions[y[i]][1]:
                correct_predictions[y[i]][0] = probabilities[i].argmax(axis=-1)
                correct_predictions[y[i]][1] = probabilities[i][probabilities[i].argmax(axis=-1)]
                correct_predictions[y[i]][2] = probabilities[i][y[i]]
                correct_prediction_images[y[i]] = x[i]

    # Plotting the correct predictions and their percentages, the correct labels and the images of the most correctly
    # predicted labels, one for each digit
    fig = plt.figure("Most Correct Predictions for Each Digit", figsize=(20, 10))
    for i in range(len(correct_predictions)):
        ax = fig.add_subplot(5, 2, i + 1)
        image = correct_prediction_images[i]
        true_label = i
        false_label = correct_predictions[i][0]
        false_prob = correct_predictions[i][1]
        true_prob = correct_predictions[i][2]
        if not np.array_equal(correct_prediction_images[i], np.zeros(784)):
            ax.imshow(image.reshape(28, 28), cmap='bone')
            ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
                         f'pred label: {false_label} ({false_prob:.3f})')
        else:
            ax.set_title(f' All {i}s were incorrectly predicted')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.show()


# Function that plots images for the most incorrect predictions, one for each digit
# param x: Test samples
# param y: Labels of the test samples
# param probabilities: Probabilities for every prediction that the network made for the labels of the test samples
def show_most_incorrect_predictions(x, y, probabilities):
    # Array that holds the probabilities and predictions for the most incorrect predictions, one for each digit
    false_predictions = np.zeros((10, 3))

    # Array that holds the images whose predicted labels are among the most incorrect predictions.
    false_prediction_images = np.zeros((10, 784))

    # Finding the most incorrect predictions, one for each digit
    for i in range(len(x)):
        if probabilities[i].argmax(axis=-1) != y[i]:
            if probabilities[i][probabilities[i].argmax(axis=-1)] > false_predictions[y[i]][1]:
                false_predictions[y[i]][0] = probabilities[i].argmax(axis=-1)
                false_predictions[y[i]][1] = probabilities[i][probabilities[i].argmax(axis=-1)]
                false_predictions[y[i]][2] = probabilities[i][y[i]]
                false_prediction_images[y[i]] = x[i]

    # Plotting the false predictions and their percentages, the correct labels and the predictions for those labels and
    # the images of the most incorrectly predicted labels, one for each digit
    fig = plt.figure("Most Incorrect Predictions for Each Digit", figsize=(20, 10))
    for i in range(len(false_predictions)):
        ax = fig.add_subplot(5, 2, i + 1)
        image = false_prediction_images[i]
        true_label = i
        false_label = false_predictions[i][0]
        false_prob = false_predictions[i][1]
        true_prob = false_predictions[i][2]
        if not np.array_equal(false_prediction_images[i], np.zeros(784)):
            ax.imshow(image.reshape(28, 28), cmap='bone')
            ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
                         f'pred label: {false_label} ({false_prob:.3f})')
        else:
            ax.set_title(f' All {i}s were correctly predicted')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.show()
