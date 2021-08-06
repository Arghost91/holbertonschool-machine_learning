#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
    * n is the number of data points
    * d is the number of dimensions for each data point
    * k is a positive integer containing the number of clusters
    * iterations is a positive integer containing the maximum
    number of iterations that should be performed
    * If no change in the cluster centroids occurs between
    iterations, your function should return
    * Initialize the cluster centroids using a multivariate
    uniform distribution (based on0-initialize.py)
    * If a cluster contains no data points during the update
    step, reinitialize its centroid
    * Returns: C, clss, or None, None on failure
    * C is a numpy.ndarray of shape (k, d) containing the
    centroid means for each cluster
    * clss is a numpy.ndarray of shape (n,) containing the
    index of the cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    if iterations <= 0 or iterations is not int:
        return None
    n, d = X.shape
    cent = np.random.uniform(np.min(X, axis=0),
                              np.max(X, axis=0),
                              size=(k, d))
    for i in range(iterations):
        cent_copy = np.ndarray.copy(cent)
        near = X - cent_copy[:, np.newaxis]
        dist = np.sqrt((near ** 2).sum(axis=2))
        clss = np.argmin(dist, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                cent[j] = np.random.uniform(np.min(X, axis=0),
                                            np.max(X, axis=0),
                                            size=(1, d))
            else:
                cent[j] = (X[clusters == j]).mean(axis=0)
        near = X - cent[:, np.newaxis]
        dist = np.sqrt((near ** 2).sum(axis=2))
        clss = np.argmin(dist, axis=0)
        if np.all(cent_copy == cent):
            return cent, clss
    return cent, clss
