#!/usr/bin/env python3

import numpy as np
"""
Define a neural network with one hidden layer performing binary classification
"""


class NeuralNetwork:
    """
    Class that define a neural network with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
