#!/usr/bin/env python3
"""
Function that builds a neural network with the Keras library
"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    * nx is the number of input features to the network
    * layers is a list containing the number of nodes in each layer of the network
    * activations is a list containing the activation functions used for each layer of the network
    * lambtha is the L2 regularization parameter
    * keep_prob is the probability that a node will be kept for dropout
    * You are not allowed to use the Input class
    * Returns: the keras model
    """
    model = k.Sequential()
    k_regularizer = k.regularizers.l2(lambtha)
    for i in range(len(layers)):
        model.add(k.layers.Dense(layers[i], input_dim=nx, activation=activations[i],
                                kernel_regularizer=k_regularizer))
        if i < (len(layers) - 1):
            model.add(k.layers.Dropout(1 - keep_prob))
    return model
