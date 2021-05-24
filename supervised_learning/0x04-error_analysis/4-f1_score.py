#!/usr/bin/env python3
"""
Function that calculates the F1 score of a confusion matrix
"""
import numpy as np


def f1_score(confusion):
    """
    * confusion is a confusion numpy.ndarray of shape (classes, classes) where row
    indices represent the correct labels and column indices represent the predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the F1 score of each class
    """
    TP = np.diagonal(confusion)
    FP = confusion.sum(axis=0) - np.diagonal(confusion)
    FN = confusion.sum(axis=1) - np.diagonal(confusion)
    F1 = (2 * TP) / ((2 * TP) + FP + FN)
    return F1
