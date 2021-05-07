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
        return self.__L
    @property
    def cache(self):
        return self.__cache
    @property
    def weights(self):
        return self.__weights
    
    def forward_prop(self, X):
        self.__cache["A0"] = X
        for i in range(self.__L):
            r = np.dot(self.__weights["W" + str(i+1)], self.cache["A" + str(i)]) + self.__weights["b" + str(i+1)]
            self.__cache["A" + str(i+1)] = 1 / (1 + np.exp(-r))
        return self.__cache["A" + str(self.__L)], self.__cache
      
    def cost(self, Y, A):
        m = len(Y[0])
        return (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1.0000001 - A)))
      
    def evaluate(self, X, Y):
        self.forward_prop(X)
        pred = np.where(self.__cache["A" + str(self.__L)] >= 0.5, 1, 0)
        cost = self.cost(Y, self.__cache["A" + str(self.__L)])
        return pred, cost
      
    def gradient_descent(self, Y, cache, alpha=0.05):
        m = len(Y[0])
        dr = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A, A_prev = cache['A' + str(i)], cache['A' + str(i-1)]
            W = self.weights["W" + str(i)]
            dW = (1 / m) * np.dot(dr, A_prev.T)
            db = (1 / m) * np.sum(dr, axis=1, keepdims=True)
            dr = np.dot(W.T, dr) * (A * (1 - A))
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db