#!/usr/bin/env python3
"""
Functions that saves and loads a model’s configuration
in JSON format
"""
import tensorflow.keras as k


def save_config(network, filename):
    """
    * network is the model whose configuration should be saved
    * filename is the path of the file that the configuration
    should be saved to
    * Returns: None
    """
    network_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(network_json)
    return None


def load_config(filename):
    """
    * filename is the path of the file containing the model’s
    configuration in JSON format
    * Returns: the loaded model
    """
    with open(filename, 'r') as json_file:
        json_load = json_file.read()
    return k.models.model_from_json(json_load)
