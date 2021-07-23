#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    * matrix is a numpy.ndarray of shape (n, n) whose definiteness should be calculated
    * If matrix is not a numpy.ndarray, raise a TypeError with the
    message matrix must be a numpy.ndarray
    * If matrix is not a valid matrix, return None
    * Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite if the
    matrix is positive definite, positive semi-definite, negative
    semi-definite, negative definite of indefinite, respectively
    * If matrix does not fit any of the above categories, return None
    * You may import numpy as np
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not (matrix == matrix.T).all():
        return None
    eigenvalues = np.linalg.eig(matrix)
    posit = 0
    negat = 0
    semi = 0
    for i in eigenvalues:
        if i > 0:
            posit = 1
        if i < 0:
            negat = 1
        if i == 0:
            semi = 0
    if posit and not negat and not semi:
        return "Positive definite"
    elif not posit and not semi and negat:
        return "Negative definite"
    elif posit and negat and not semi:
        return "Indefinite"
    elif posit and not negat and semi:
        return "Positive semi-definite"
    elif not posit and negat and semi:
        return "Negative semi-definite"
    else:
        return None
