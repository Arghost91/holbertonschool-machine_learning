#!/usr/bin/env python3
"""
Function that normalizes an unactivated output of
a neural network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    * Z is a numpy.ndarray of shape (m, n) that should be normalized
        * m is the number of data points
        * n is the number of features in Z
    * gamma is a numpy.ndarray of shape (1, n) containing the scales used for batch normalization
    * beta is a numpy.ndarray of shape (1, n) containing the offsets used for batch normalization
    * epsilon is a small number used to avoid division by zero
    * Returns: the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / (((var ** 2) + epsilon) ** (1 / 2))
    Z_mat = gamma * Z_norm + beta
    return Z_mat
    
