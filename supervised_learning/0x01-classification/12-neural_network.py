#!/usr/bin/env python3

import numpy as np

class NeuralNetwork:
    
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0
    @property
    def W1(self):
        return self.__W1
    @property
    def b1(self):
        return self.__b1
    @property
    def A1(self):
        return self.__A1
    @property
    def W2(self):
        return self.__W2
    @property
    def b2(self):
        return self.__b2
    @property
    def A2(self):
        return self.__A2
      
    def forward_prop(self, X):
        r1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-r1))
        r2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-r2))
        return self.__A1, self.__A2
    
    def cost(self, Y, A):
        m = len(Y[0])
        return (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1.0000001 - A)))
    
    def evaluate(self, X, Y):
        A2 = self.forward_prop(X)
        pred = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return pred, cost
