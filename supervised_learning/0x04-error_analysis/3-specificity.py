#!/usr/bin/env python3
"""
Function that calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    * confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the specificity
    of each class
    """
    FP = confusion.sum(axis=0) - np.diagonal(confusion)
    FN = confusion.sum(axis=1) - np.diagonal(confusion)
    TP = np.diagonal(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    TNR = TN / (TN + FP)
    return TNR
