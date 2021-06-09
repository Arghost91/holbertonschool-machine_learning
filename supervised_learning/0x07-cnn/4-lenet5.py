#!/usr/bin/env python3
"""
Function that builds a modified version of the LeNet-5
architecture using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    * x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        * m is the number of images
    * y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
    the network
    * The model should consist of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes
    * All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
    tf.contrib.layers.variance_scaling_initializer()
    * All hidden layers requiring activation should use the relu activation
    function
    * Returns:
        * a tensor for the softmax activated output
        * a training operation that utilizes Adam optimization (with default
        hyperparameters)
        * a tensor for the loss of the netowrk
        * a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    conv_6 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                              activation=activation, kernel_initializer=init)(x)
    pool_6 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_6)
    conv_16 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                               activation=activation,
                               kernel_initializer=init)(pool_6)
    pool_16 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_16)
    flatten = tf.layers.Flatten()(pool_16)
    fc_120 = tf.layers.Dense(units=120, activation=activation,
                             kernel_initializer=init)(flatten)
    fc_84 = tf.layers.Dense(units=84, activation=activation,
                            kernel_initializer=init)(fc_120)
    fc_10 = tf.layers.Dense(units=10, activation=None,
                            kernel_initializer=init)(fc_84)

    # Softmax
    y_pred = tf.nn.softmax(fc_10)

    # Accuracy
    y_max = tf.argmax(y, 1)
    y_pred_max = tf.argmax(fc_10, 1)
    evaluation = tf.equal(y_max, y_pred_max)
    accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, fc_10)

    # Train
    train = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train, loss, accuracy
