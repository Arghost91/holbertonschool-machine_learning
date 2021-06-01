#!/usr/bin/env python3
"""
Function to also analyze validaiton data
"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    * validation_data is the data to validate the
    model with, if not None
    """
    train = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                        validation_data=validation_data, verbose=verbose,
                        shuffle=shuffle)
    return train
