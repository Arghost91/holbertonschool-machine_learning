#!/usr/bin/env python3
"""
Function that builds a dense block as described
in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    * X is the output from the previous layer
    * nb_filters is an integer representing the number of filters in X
    * growth_rate is the growth rate for the dense block
    * layers is the number of layers in the dense block
    * You should use the bottleneck layers used for DenseNet-B
    * All weights should use he normal initialization
    * All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    * Returns: The concatenated output of each layer within the Dense
    Block and the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()
    activ = "relu"

    for i in range(layers):
        normal_1 = K.layers.BatchNormalization()(X)
        activ_1 = K.layers.Activation(activ)(normal_1)
        filt = 4 * growth_rate
        bottleneck = K.layers.Conv2D(filters=filt, kernel_size=(1, 1),
                                     padding='same', kernel_initializer=init)(activ_1)

        normal_2 = K.layers.BatchNormalization()(bottleneck)
        activ_2 = K.layers.Activation(activ)(normal_2)
        conv = K.layers.Conv2D(filters=grouth_rate, kernel_size=(3, 3),
                               padding='same', kernel_initializer=init)(activ_2)

        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate
return X, nb_filters
