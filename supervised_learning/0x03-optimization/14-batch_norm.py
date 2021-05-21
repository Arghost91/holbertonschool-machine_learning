
#!/usr/bin/env python3
"""
Function that creates a batch normalization
layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    * prev is the activated output of the previous layer
    * n is the number of nodes in the layer to be created
    * activation is the activation function that should
      be used on the output of the layer
    * you should use the tf.layers.Dense layer as the base
      layer with kernal initializer
      tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    * your layer should incorporate two trainable parameters,
      gamma and beta, initialized as vectors of 1 and 0 respectively
    * you should use an epsilon of 1e-8
    * Returns: a tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    out = tf.layers.Dense(n, activation, kernel_initializer = init)
    x = out(prev)
    gamma = tf.Variable(initial_value=tf.constant(1, shape=[n], name="gamma"))
    beta = tf.Variable(initial_value=tf.constant(0, shape=[n], name="beta"))
    mean, var = tf.nn.moments(x, axes=0)
    Z = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    return Z
