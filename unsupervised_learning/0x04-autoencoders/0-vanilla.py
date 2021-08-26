#!/usr/bin/env python3
"""
Function that creates an autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    * input_dims is an integer containing the dimensions of the model input
    * hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
    * latent_dims is an integer containing the dimensions of the latent space
    representation
    * Returns: encoder, decoder, auto
        * encoder is the encoder model
        * decoder is the decoder model
        * auto is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    * All layers should use a relu activation except for the last layer in
    the decoder, which should use sigmoid
    """
    num = len(hidden_layers)
    input_img = keras.Input(shape=(input_dims,))
    for i in range(num):
        encoded = keras.layers.Dense(hidden_layers[i], activation='relu')(input_img)
    h = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs=input_img, outputs=h)

    decoded = keras.Input(shape=(latent_dims,))
    for j in range(num - 1, -1, -1):
        dec = keras.layers.Dense(hidden_layers[j], activation='relu')(decoded)
    r = keras.layers.Dense(input_dims, activation='sigmoid')(dec)
    decoder = keras.Model(inputs=decoded, outputs=r)

    output = decoder(encoder(input_img))
    auto = keras.Model(inputs=input_img, outputs=output)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
