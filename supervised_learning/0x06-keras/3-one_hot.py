#!/usr/bin/env python3
"""
Function that converts a label vector into a one-hot matrix
"""
import tensorflow.keras as k


def one_hot(labels, classes=None):
    """
    * The last dimension of the one-hot matrix must be the
    number of classes
    * Returns: the one-hot matrix
    """
    one = k.utils.to_categorical(labels, num_classes=classes)
    return one
