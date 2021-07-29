#!/usr/bin/env python3
"""
Function that calculates the intersection of obtaining
this data with the various hypothetical probabilities
"""
import numpy as np


def likelihood(x, n, P):
    """
    * x is the number of patients that develop severe side effects
    * n is the total number of patients observed
    * P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    * If n is not a positive integer, raise a ValueError with the
    message n must be a positive integer
    * If x is not an integer that is greater than or equal to 0,
    raise a ValueError with the message x must be an integer that
    is greater than or equal to 0
    * If x is greater than n, raise a ValueError with the message
    x cannot be greater than n
    * If P is not a 1D numpy.ndarray, raise a TypeError with the
    message P must be a 1D numpy.ndarray
    * If any value in P is not in the range [0, 1], raise a ValueError
    with the message All values in P must be in the range [0, 1]
    * Returns: a 1D numpy.ndarray containing the likelihood of
    obtaining the data, x and n, for each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    err = "x must be an integer that is greater than or equal to 0"
    if type(x) is not int or x < 0:
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for i in P:
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    likelihood = np.math.factorial(n) / (np.math.factorial(x) *
                                         np.math.factorial(n - x))
    likelihood = likelihood * ((P ** x) * (1 - P) ** (n - x))
    return likelihood


def intersection(x, n, P, Pr):
    """
    * x is the number of patients that develop severe side effects
    * n is the total number of patients observed
    * P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    * Pr is a 1D numpy.ndarray containing the prior beliefs of P
    * If n is not a positive integer, raise a ValueError with the
    +message n must be a positive integer
    * If x is not an integer that is greater than or equal to 0,
    raise a ValueError with the message x must be an integer that
    is greater than or equal to 0
    * If x is greater than n, raise a ValueError with the message
    x cannot be greater than n
    * If P is not a 1D numpy.ndarray, raise a TypeError with the
    message P must be a 1D numpy.ndarray
    * If Pr is not a numpy.ndarray with the same shape as P, raise
    a TypeError with the message Pr must be a numpy.ndarray with the
    same shape as P
    * If any value in P or Pr is not in the range [0, 1], raise a
    ValueError with the message All values in {P} must be in the
    range [0, 1] where {P} is the incorrect variable
    * If Pr does not sum to 1, raise a ValueError with the message
    Pr must sum to 1 Hint: use numpy.isclose
    * All exceptions should be raised in the above order
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    err = "x must be an integer that is greater than or equal to 0"
    if type(x) is not int or x < 0:
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    err_2 = "Pr must be a numpy.ndarray with the same shape as P"
    if type(Pr) is not np.ndarray or len(Pr.shape) != len(P.shape):
        raise TypeError(err_2)
    for i, j in zip(P, Pr):
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        if j < 0 or j > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    intersec = likelihood(x, n, P) * Pr
    return intersec
