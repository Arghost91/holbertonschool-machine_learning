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
    sub_mean = [i - mean for i in X]
    num = sum([sub_mean[i] for i in len(sub_mean)])
    cov = num / (n - 1)
    return mean, cov    
