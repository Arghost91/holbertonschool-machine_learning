#!/usr/bin/env python3
"""
Function that calculates the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    * y is a placeholder for the labels of the input data
    * y_pred is a tensor containing the network’s predictions
    * Returns: a tensor containing the decimal accuracy of the prediction
    """
    y_max = tf.argmax(y, 1)
    y_pred_max = tf.argmax(y_pred, 1)
    evaluation = tf.equal(y_max, y_pred_max)
    accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))
    return accuracy
