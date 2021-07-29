#!/usr/bin/env python3
"""
Function that calculates the cost of the t-SNE transformation
"""
import numpy as np


def cost(P, Q):
    """
    * P is a numpy.ndarray of shape (n, n) containing the P affinities
    * Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    * Returns: C, the cost of the transformation
    """
    PQ = np.divide(P, Q)
    cost = np.sum(P * np.log(PQ))
    return cost
