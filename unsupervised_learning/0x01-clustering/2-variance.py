  
#!/usr/bin/env python3
"""
Function that calculates the total intra-cluster variance
for a data set
"""
import numpy as np


def variance(X, C):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    * Returns: var, or None on failure
    * var is the total variance
    """
    if type(X) is not np.ndarray or len(X) != 2:
        return None
    if type(C) is not np.ndarray or len(X) != 2:
        return None
    k, d = C.shape
    if d != X.shape[1]:
        return None
    if type(k) is not int or k <= 0:
        return None
    delt = X[:, np.newaxis] - C
    dist = np.sqrt((del ** 2).sum(axis=2))
    mini = np.min(dist, axis=0)
    car = np.sum(min ** 2)
    return var
