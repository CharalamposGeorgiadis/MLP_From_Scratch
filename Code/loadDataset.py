import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Function that loads the dataset
# return: Training samples, labels of the training samples, test samples, labels of the test samples
def load_dataset():
    # Loading the MNIST database of handwritten digits
    # Training set
    train_set = pd.read_csv('dataset/mnist_train.csv')
    x_train = train_set.drop(columns=['label'])
    y_train = train_set['label'].values

    # Test set
    test_set = pd.read_csv('dataset/mnist_test.csv')
    x_test = test_set.drop(columns=['label'])
    y_test = test_set['label'].values

    # Normalizing the data
    x_train = np.array(pd.DataFrame(MinMaxScaler().fit_transform(x_train)))
    x_test = np.array(pd.DataFrame(MinMaxScaler().fit_transform(x_test)))
    return x_train, y_train, x_test, y_test
