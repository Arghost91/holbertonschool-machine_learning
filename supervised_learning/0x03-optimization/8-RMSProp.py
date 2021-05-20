#!/usr/bin/env python3
"""
Function that updates a variable using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    * loss is the loss of the network
    * alpha is the learning rate
    * beta2 is the RMSProp weight
    * epsilon is a small number to avoid division by zero
    * Returns: the RMSProp optimization operation
    """
    rms = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    return rms.minimize(loss)
