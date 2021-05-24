#!/usr/bin/env python3
"""
Function that calculates the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    * confusion is a confusion numpy.ndarray of shape (classes, classes) where row
    indices represent the correct labels and column indices represent the predicted labels
        *classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of each class
    """
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=1)
    TPR = TP / P
    return TPR
