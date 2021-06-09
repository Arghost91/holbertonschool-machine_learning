#!/usr/bin/env python3
"""
Function that builds a modified version of the LeNet-5
architecture using keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    * X is a K.Input of shape (m, 28, 28, 1) containing the input images for
    the network
        * m is the number of images
    * The model should consist of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes
    * All layers requiring initialization should initialize their kernels with
    the he_normal initialization method
    * All hidden layers requiring activation should use the relu activation
    function
    * Returns: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal()
    activation = K.activations.relu

    conv_6 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=activation, kernel_initializer=init)(X)
    pool_6 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_6)
    conv_16 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                              activation=activation,
                              kernel_initializer=init)(pool_6)
    pool_16 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_16)
    flatten = K.layers.Flatten()(pool_16)
    fc_120 = K.layers.Dense(units=120, activation=activation,
                            kernel_initializer=init)(flatten)
    fc_84 = K.layers.Dense(units=84, activation=activation,
                           kernel_initializer=init)(fc_120)
    fc_10 = K.layers.Dense(units=10, activation=None,
                           kernel_initializer=init)(fc_84)

    network = K.models.Model(inputs=X, outputs=fc_10)
    # Train
    optim = K.optimizers.Adam()

    network.compile(optimizer=optim, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return network
