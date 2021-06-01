#!/usr/bin/env python3
"""
Functions that saves and loads an entire model
"""
import tensorflow.keras as k


def save_model(network, filename):
    """
    * network is the model to save
    * filename is the path of the file that the model should be saved to
    * Returns: None
    """
    k.models.save_model(network, filename)
    return None


def load_model(filename):
    """
    * filename is the path of the file that the model should be loaded from
    * Returns: the loaded model
    """
    return k.models.load_model(filename)
