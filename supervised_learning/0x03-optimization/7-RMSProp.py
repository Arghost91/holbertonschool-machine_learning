#!/usr/bin/env python3
"""
Function that updates a variable using the RMSProp optimization algorithm
"""
import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    * alpha is the learning rate
    * beta2 is the RMSProp weight
    * epsilon is a small number to avoid division by zero
    * var is a numpy.ndarray containing the variable to be updated
    * grad is a numpy.ndarray containing the gradient of var
    * s is the previous second moment of var
    * Returns: the updated variable and the new moment, respectively
    """
    S_d = (beta2 * S_d) + ((1 - beta2) * grad ** 2) 
    W = W - (alpha * (grad / (epsilon + (S_d ** (1 / 2)))))
    return W, S_d
