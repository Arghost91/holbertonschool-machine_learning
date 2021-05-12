#!/usr/bin/env python3

import numpy as np
"""
Defines a deep neural network performing binary classification
"""


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            W = 'W' + str(i+1)
            layers_1 = (layers[i - 1])
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights[W] = np.random.randn(layers[i], 
                                                  nx) * np.sqrt(2 / nx)
            else:
                sq2 = np.sqrt(2 / (layers_1)
                self.weights[W] = np.random.randn(layers[i], 
                                                  layers_1) * sq2)
            self.weights['b' + str(i+1)] = np.zeros((layers[i], 1))
