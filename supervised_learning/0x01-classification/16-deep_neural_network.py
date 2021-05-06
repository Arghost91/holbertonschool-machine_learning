#!/usr/bin/env python3

import numpy as np

class DeepNeuralNetwork:
    
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
        if layers[0] < 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(1, self.L):
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights['W' + str(i)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / (layers[i - 1]))
            self.weights['b' + str(i)] = np.zeros((layers[i], 1))
