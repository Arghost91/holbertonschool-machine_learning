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
            W = "W" + str(i+1)
            A = "A" + str(i)
            b1 = "b" + str(i+1)
            r = np.dot(self.__weights[W], self.cache[A]) + self.__weights[b1]
            self.__cache["A" + str(i+1)] = 1 / (1 + np.exp(-r))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = len(Y[0])
        log = np.log(1.0000001 - A)
        su = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), log)
        return (-1 / m) * np.sum(su)

    def evaluate(self, X, Y):
        """
        Evaluates the neural network???s predictions
        """
        self.forward_prop(X)
        pred = np.where(self.__cache["A" + str(self.__L)] >= 0.5, 1, 0)
        cost = self.cost(Y, self.__cache["A" + str(self.__L)])
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = len(Y[0])
        dr = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i-1)]
            W = self.weights["W" + str(i)]
            dW = (1 / m) * np.dot(dr, A_prev.T)
            db = (1 / m) * np.sum(dr, axis=1, keepdims=True)
            dr = np.dot(W.T, dr) * (A_prev * (1 - A_prev))
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db
