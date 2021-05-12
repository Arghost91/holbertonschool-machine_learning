#!/usr/bin/env python3

import numpy as np
"""
Converts a a one-hot matrix into a vector of labels
"""


def one_hot_decode(one_hot):
    """
    Class that converts a numeric label vector into a one-hot matrix
    """
    if type(one_hot) is not np.ndarray:
        return None
    else:
        return np.argmax(one_hot, axis=0)
