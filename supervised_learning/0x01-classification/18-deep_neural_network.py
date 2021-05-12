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
            W = 'W' + str(i+1)
            layers_1 = (layers[i - 1])
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights[W] = np.random.randn(layers[i],
                                                  nx) * np.sqrt(2 / nx)
            else:
                sq2 = np.sqrt(2 / (layers_1))
                self.weights[W] = np.random.randn(layers[i],
                                                  layers_1) * sq2
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            A = "A" + str(i)
            b1 = "b" + str(i+1)
            r = np.dot(self.__weights["W" + str(i+1)], self.cache[A]) + self.__weights["b1]
            self.__cache["A" + str(i+1)] = 1 / (1 + np.exp(-r))
        return self.__cache["A" + str(self.__L)], self.__cache
