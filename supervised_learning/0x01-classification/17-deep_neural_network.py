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
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights['W' + str(i+1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i+1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / (layers[i - 1]))
            self.weights['b' + str(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        getter function
        """
        return self.__L

    @property
    def cache(self):
        """
        getter function
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter function
        """
        return self.__weights
