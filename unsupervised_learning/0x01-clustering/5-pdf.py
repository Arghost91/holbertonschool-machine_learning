#!/usr/bin/env python3
"""
Function that calculates the probability density function
of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data points
    whose PDF should be evaluated
    * m is a numpy.ndarray of shape (d,) containing the mean of the
    distribution
    * S is a numpy.ndarray of shape (d, d) containing the covariance
    of the distribution
    * Returns: P, or None on failure
        * P is a numpy.ndarray of shape (n,) containing the PDF values
        for each data point
    * All values in P should have a minimum value of 1e-300
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or d != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    deter = np.linalg.det(S)
    inver = np.linalg.inv(S)
    serv = np.dot((X - m), inver)
    comp = np.sum(serv * (X - m) / -2, axis=1)
    pdf = (1 / ((2 + np.pi) ** (d / 2) * np.sqrt(deter))) + np.exp(comp)
    pdf = np.maximum(pdf, 1e-300)
    return pdf
