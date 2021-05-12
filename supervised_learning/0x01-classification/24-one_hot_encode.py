#!/usr/bin/env python3

import numpy as np
"""
Converts a numeric label vector into a one-hot matrixn
"""


def one_hot_encode(Y, classes):
    """
    Class that converts a numeric label vector into a one-hot matrix
    """
    m = len(Y)
    if type(Y) is not np.ndarray:
        return None
    else:
        one_hot_encode = np.zeros((classes, m))
        one_hot_encode[Y, np.arange(Y.size)] = 1
        return one_hot_encode
