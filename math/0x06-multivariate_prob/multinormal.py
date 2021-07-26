#!/usr/bin/env python3
"""
Class that represents a Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """
    Class that represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        * data is a numpy.ndarray of shape (d, n) containing the data set:
        * n is the number of data points
        * d is the number of dimensions in each data point
        * If data is not a 2D numpy.ndarray, raise a TypeError with the
        message data must be a 2D numpy.ndarray
        * If n is less than 2, raise a ValueError with the message data
        must contain multiple data points
        """
        if type(data) is not np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("X must contain multiple data points")
        me = np.mean(data, axis=1)
        me = np.reshape(me, (d, 1))
        self.mean = me
        sub_mean = data - self.mean
        self.cov = np.dot(sub_mean.T, sub_mean) / (n - 1)
