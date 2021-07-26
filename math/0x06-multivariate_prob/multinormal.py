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
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1)
        self.mean = np.reshape(self.mean, (d, 1))
        sub_mean = data - self.mean
        self.cov = np.dot(sub_mean, sub_mean.T) / (n - 1)

    def pdf(self, x):
        """
        * x is a numpy.ndarray of shape (d, 1) containing the data point
        whose PDF should be calculated
            * d is the number of dimensions of the Multinomial instance
        * If x is not a numpy.ndarray, raise a TypeError with the message
        x must be a numpy.ndarray
        * If x is not of shape (d, 1), raise a ValueError with the message
        x must have the shape ({d}, 1)
        * Returns the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = x.shape[0]
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        sub_mean = x - self.mean
        det = np.linalg.det(self.cov)
        sol = np.linalg.solve(self.cov, sub_mean)
        pdf = (1. / (np.sqrt((2 * np.pi)**d * det)) *
               np.exp(-(sol.T.dot(sub_mean)) / 2))
        return pdf[0][0]
