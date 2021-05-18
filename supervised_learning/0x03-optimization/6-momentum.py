#!/usr/bin/env python3
"""
Function that creates the training operation for a neural network
in tensorflow using the gradient descent with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
     """
    * loss is the loss of the network
    * alpha is the learning rate
    * beta1 is the momentum weight
    * Returns: the momentum optimization operation
    """
    op = tf.train.MomentumOptimizer(alpha, beta1)
    return op.minimize(loss)
