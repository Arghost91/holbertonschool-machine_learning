#!/usr/bin/env python3
"""
Function that updates the weights and biases of a neural
network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    * Y is a one-hot numpy.ndarray of shape (classes, m) that
    contains the correct labels for the data
        * classes is the number of classes
        * m is the number of data points
    * weights is a dictionary of the weights and biases of the neural network
    * cache is a dictionary of the outputs of each layer of the neural network
    * alpha is the learning rate
    * lambtha is the L2 regularization parameter
    * L is the number of layers of the network
    * The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    * The weights and biases of the network should be updated in place
    """
    m = len(Y[1])
    for i in reversed(range(L)):
        A_prev = cache['A' + str(i+1)]
        A_dw = cache['A' + str(i)]
        if i == L - 1:
            dr = A_prev - Y
            W = weights["W" + str(i+1)]
        else:
            dr = np.dot(W.T, dr) * (1 - np.power(A_prev, 2))
            W = weights["W" + str(i+1)]
        dW = (1 / m) * np.dot(dr, A_dw.T) + ((lambtha / m) *
                                               weights["W" + str(i+1)])
        db = (1 / m) * np.sum(dr, axis=1, keepdims=True)
        weights["W" + str(i+1)] = weights["W" + str(i+1)] - alpha * dW
        weights["b" + str(i+1)] = weights["b" + str(i+1)] - alpha * db
