#!/usr/bin/env python3
"""
Function that updates a variable in place using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    * alpha is the learning rate
    * beta1 is the weight used for the first moment
    * beta2 is the weight used for the second moment
    * epsilon is a small number to avoid division by zero
    * var is a numpy.ndarray containing the variable to be updated
    * grad is a numpy.ndarray containing the gradient of var
    * v is the previous first moment of var
    * s is the previous second moment of var
    * t is the time step used for bias correction
    * Returns: the updated variable, the new first moment, and the new second moment, respectively
    """
    V_dw = (beta1 * v) + ((1 - beta1) * grad)
    S_dw = (beta2 * s) + ((1 - beta2) * grad ** 2)
    V_corrected = V_dw / (1 - (beta1 ** t))
    S_corrected = S_dw / (1 - (beta2 ** t))
    var = var - (alpha * (V_corrected / ((S_corrected ** (1 / 2)) + epsilon)))
    return var, V_dw, S_dw
