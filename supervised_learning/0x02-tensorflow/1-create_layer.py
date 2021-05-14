#!/usr/bin/env python3
"""
Tensor output of the layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Function that return the tensor output of the layer
    """
    initial = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initial,
                            name='layer')
    return layer(prev)
