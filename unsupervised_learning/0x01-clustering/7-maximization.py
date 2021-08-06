#!/usr/bin/env python3
"""
Function that calculates the maximization step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    * Returns: pi, m, S, or None, None, None on failure
        * pi is a numpy.ndarray of shape (k,) containing the updated
        priors for each cluster
        * m is a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
        * S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    if n != g.shape[1]:
        return None, None, None
    clust = np.sum(g, axis=0)
    clust = np.sum(clust)
    if int(clust) != X.shape[0]:
        return None, None, None
    k = g.shape[0]
    post = np.sum(g, axis=1)
    pi = post / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        m[i] = np.matmul(g[i], X) / post[i]
        S[i] = np.matmul(g[i] * (X - m[i]).T, (X - m[i])) / post[i]
    return pi, m, S
