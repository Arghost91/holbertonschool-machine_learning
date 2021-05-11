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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = len(Y[0])
        return (-1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply((1 - Y), np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        A = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = len(Y[0])
        dW = (1 / m) * (np.dot(X, (A - Y).T))
        db = (1 / m) * (np.sum(A - Y))
        self.__W -= (alpha * (dW.T))
        self.__b -= (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron by updating the private attributes __W, __b, and __A
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("iterations must be a float")
        if alpha <= 0:
            raise ValueError("iterations must be a positive")
        list_cost = []
        list_iteration = []
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        for iteration in range(iterations+1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            cost = self.cost(Y, A)
            if verbose:
                if iteration % step == 0:
                    print("Cost after {} iterations: {}".format(iteration,
                                                                cost))
                    list_cost.append(cost)
                    list_iteration.append(iteration)
        if graph:
            plt.plot(list_iteration, list_cost, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
        return self.evaluate(X, Y)
