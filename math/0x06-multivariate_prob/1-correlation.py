#!/usr/bin/env python3
"""
Function that calculates a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    * C is a numpy.ndarray of shape (d, d) containing a covariance matrix
        * d is the number of dimensions
        * If C is not a numpy.ndarray, raise a TypeError with the message
        C must be a numpy.ndarray
        * If C does not have shape (d, d), raise a ValueError with the message
        C must be a 2D square matrix
    * Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1] or len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    diag = np.diag(C)
    stan_dev = np.sqrt(diag)
    out_stan = np.outer(stan_dev, stan_dev)
    corr = C / out_stan
    return corr
