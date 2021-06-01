  
#!/usr/bin/env python3
"""
Function that builds a neural network with the Keras library
"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    * nx is the number of input features to the network
    * layers is a list containing the number of nodes in each layer of the network
    * activations is a list containing the activation functions used for each layer of the network
    * lambtha is the L2 regularization parameter
    * keep_prob is the probability that a node will be kept for dropout
    * You are not allowed to use the Sequential class
    * Returns: the keras model
    """
    inputs = k.Input(shape=(nx,))
    k_regularizer = k.regularizers.l2(lambtha)
    x = k.layers.Dense(layers[0], input_dim=nx,
                       activation=activations[0],
                       kernel_regularizer=k_regularizer)(inputs)
    for i in range(1, len(layers)):
        dropout = k.layers.Dropout(1 - keep_prob)(x)
        x = k.layers.Dense(layers[i], input_dim=nx,
                           activation=activations[i],
                           kernel_regularizer=k_regularizer)(dropout)
        model = k.Model(inputs, x)
    return model
