#!/usr/bin/env python3
"""
Function that builds a transition layer as described in
Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    * X is the output from the previous layer
    * nb_filters is an integer representing the number of filters in X
    * compression is the compression factor for the transition layer
    * Your code should implement compression as used in DenseNet-C
    * All weights should use he normal initialization
    * All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    * Returns: The output of the transition layer and the number
    of filters within the output, respectively
    """
    init = K.initializers.he_normal()
    activ = "relu"
    filt = int(nb_filters * compression)

    normal_1 = K.layers.BatchNormalization()(X)
    activ_1 = K.layers.Activation(activ)(normal_1)
    conv_1 = K.layers.Conv2D(filters=filt, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=init)(activ_1)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         padding='same')(conv_1)
    return avg_pool, filt
