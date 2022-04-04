# MLP_From_Scratch
***NOTE: Extract the "dataset.zip" file before running the code!***

Python project that trains and evaluates a Multilayer Perceptron using only NumPy on the MNIST dataset.

The code was based on the one provided in the following link: 
https://towardsdatascience.com/building-an-artificial-neural-network-using-pure-numpy-3fe21acc5815

The dataset is loaded using the Pandas library and normalized using the MinMaxScaler function from scikit-learn. 

Matplotlib is used in order to display two images:
1. One shows the one image for each digit that was most confidently predicted correctly.
2. The other shows the one image for each digit that was most confidently predicted incorrectly.

108 different hyperparameters were tested in order to find the MLP that could reach the highest possible test accuracy out of these combinations.

That model was trained with the following hyperparameters:
1. Learning rate = 0.1
2. Number of hidden layeres: 4
3. Number of hidden layer neurons (500 -> 250 -> 150 -> 100)
4. Batch size = 32
5. Number of epochs = 30
6. Activation functions: SoftMax on the output layer, ReLU on all the others.



The performance of the MLP when evaluated on the whole training set and on the test set is listed below:

| Train Accuracy  | Test Accuracy | Execution TIme (Avg.) |
| ------------- | ------------- | -------------|
| 99.85% | 98.59% | ~340510 ms  |
