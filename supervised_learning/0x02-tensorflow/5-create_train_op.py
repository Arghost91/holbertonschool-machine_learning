#!/usr/bin/env python3
"""
Function that creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    * loss is the loss of the network’s prediction
    * alpha is the learning rate
    * Returns: an operation that trains the network using gradient descent
    """
    gradiente = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return gradient
