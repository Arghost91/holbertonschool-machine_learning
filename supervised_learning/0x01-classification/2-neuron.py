#!/usr/bin/env python3

import numpy as np
"""
Define a single neuron performing binary classification
"""


class Neuron:
    """
    Class that define a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function
        """
        return self.__W

    @property
    def b(self):
        """
        getter function
        """
        return self.__b

    @property
    def A(self):
        """
        getter function
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        r = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-r))
        return self.__A
