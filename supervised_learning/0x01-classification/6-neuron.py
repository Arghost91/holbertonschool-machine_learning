#!/usr/bin/env python3

import numpy as np

class Neuron:
    
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0
        
    @property
    def W(self):
        return self.__W
    @property
    def b(self):
        return self.__b
    @property
    def A(self):
        return self.__A
    
    def forward_prop(self, X):
        r = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-r))
        return self.__A
      
    def cost(self, Y, A):
        m = len(Y[0])
        return (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1.0000001 - A)))
      
    def evaluate(self, X, Y):
        A = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost
      
    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = len(Y[0])
        dW = (1 / m) * (np.dot(X, (A - Y).T))
        db = (1 / m) * (np.sum(A - Y))
        self.__W -= (alpha * (dW.T))
        self.__b -= (alpha * db)
        
    def train(self, X, Y, iterations=5000, alpha=0.05):
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("iterations must be a float")
        if alpha <= 0:
            raise ValueError("iterations must be a positive")
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)