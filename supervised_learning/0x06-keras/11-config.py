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
    json_file = open(filename, 'r')
    loaded_network_json = json_file.read()
    json_file.close()
    loaded_network = network_from_json(loaded_network_json)
    return loaded_network
