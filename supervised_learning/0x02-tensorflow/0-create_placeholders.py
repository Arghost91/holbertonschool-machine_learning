#!/usr/bin/env python3

import tensorflow as tf
"""
Returns two placeholders, x and y, for the neural network
"""


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders, x and y, for the neural network
    """
    x = tf.placeholder("float", shape=(None, nx), name='x')
    y = tf.placeholder("float", shape=(None, classes), name='y')
    return x, y
