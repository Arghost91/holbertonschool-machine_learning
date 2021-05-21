#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix
"""


def normalization_constants(X):
    """
    Function that the mean and standard deviation of each feature, respectively
    """
    return X.mean(axis=0), X.std(axis=0)
