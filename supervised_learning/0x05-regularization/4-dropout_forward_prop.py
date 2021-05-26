#!/usr/bin/env python3
"""
Function that conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    * X is a numpy.ndarray of shape (nx, m) containing the input
    data for the network
        * nx is the number of input features
        * m is the number of data points
    * weights is a dictionary of the weights and biases of the neural
    network
    * L the number of layers in the network
    * keep_prob is the probability that a node will be kept
    * All layers except the last should use the tanh activation function
    * The last layer should use the softmax activation function
    * Returns: a dictionary containing the outputs of each layer and the
    dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        W = "W" + str(i+1)
        b1 = "b" + str(i+1)
        r = np.dot(weights[W],
                cache["A" + str(i)]) + weights[b1]
        cache["A" + str(i+1)] = 1 / (1 + np.exp(-r))
        if i == L - 1:
            t = np.exp(r) / np.sum(np.exp(r), axis=0)
        else:
            t = (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r))
            d = np.random.rand(t.shape[0], t.shape[1])
            d = int(d < keep_prob)
            t = np.multiply(t, d)
            t /= keep_prob 
            cache["D" + str(i+1)] = d
        cache["A" + str(i+1)] = t
    return cache
