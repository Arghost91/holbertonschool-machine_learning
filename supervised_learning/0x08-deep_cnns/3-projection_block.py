#!/usr/bin/env python3
"""
Function that builds a projection block as described
in Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    * A_prev is the output from the previous layer
    * filters is a tuple or list containing F11, F3, F12, respectively:
        * F11 is the number of filters in the first 1x1 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F12 is the number of filters in the second 1x1 convolution
        as well as the 1x1 convolution in the shortcut connection
    * s is the stride of the first convolution in both the main path
    and the shortcut connection
    * All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    * All weights should use he normal initialization
    * Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activ = "relu"

    conv_11 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=s,
                              padding='same', kernel_initializer=init)(A_prev)
    normal_11 = K.layers.BatchNormalization()(conv_11)
    activ_11 = K.layers.Activation(activ)(normal_11)

    conv_3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                             kernel_initializer=init)(activ_11)
    normal_3 = K.layers.BatchNormalization()(conv_3)
    activ_3 = K.layers.Activation(activ)(normal_3)

    conv_12 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                              padding='same', kernel_initializer=init)(activ_3)
    normal_12 = K.layers.BatchNormalization()(conv_12)

    conv_4f = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=s, padding='same',
                              kernel_initializer=init)(A_prev)
    normal_4f = K.layers.BatchNormalization()(conv_4f)

    added = K.layers.Add()([normal_12, normal_4f])
    activat = K.layers.Activation(activ)(added)
    return activat
