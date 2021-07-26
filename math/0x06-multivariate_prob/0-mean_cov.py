#!/usr/bin/env python3
"""
Function that calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * Returns: mean, cov
    """
    n, d = X.shape
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0)
    mean = np.reshape(mean, (1, d))
    sub_mean = X - mean
    cov = np.dot(sub_mean.T, sub_mean) / (n - 1)
    return mean, cov    
