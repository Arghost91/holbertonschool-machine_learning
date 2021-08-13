#!/usr/bin/env python3
"""
Function that determines if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    * P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the standard transition matrix
        * P[i, j] is the probability of transitioning from state i to state j
        * n is the number of states in the markov chain
    * Returns: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    diag = np.diag(P)
    ab = (diag == 1)
    if ab.all():
        return True
    for i in range(len(diag)):
        for j in range(len(diag)):
            if P[i, j] > 0 and ab[j]:
                ab[i] = 1
    ab2 = (ab == 1)
    if ab2.all():
        return True
    return False
