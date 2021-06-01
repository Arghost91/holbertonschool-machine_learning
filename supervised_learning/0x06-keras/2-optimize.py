#!/usr/bin/env python3
"""
Function that sets up Adam optimization for a keras model
with categorical crossentropy loss and accuracy metrics
"""
import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """
    * network is the model to optimize
    * alpha is the learning rate
    * beta1 is the first Adam optimization parameter
    * beta2 is the second Adam optimization parameter
    * Returns: None
    """
    optim = k.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optim, loss='categorical crossentropy',
                    metrics=['accuracy'])
