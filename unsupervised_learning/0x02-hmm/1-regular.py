#!/usr/bin/env python3
"""
Function that determines the probability of a markov chain being
in a particular state after a specified number of iterations
"""
import numpy as np


def regular(P):
    """
    * P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
    * P[i, j] is the probability of transitioning from state i to state j
    * n is the number of states in the markov chain
    * Returns: a numpy.ndarray of shape (1, n) containing the steady
    state probabilities, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if (P>0).all():
        evals, evecs = np.linalg.eig(P.T)
        stat = evecs / evecs.sum()
        stat = stat[np.newaxis, :]
        return stat
