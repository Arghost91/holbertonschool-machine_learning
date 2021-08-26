#!/usr/bin/env python3
"""
Function that creates a convolutional autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    * input_dims is a tuple of integers containing the dimensions of the model input
    * filters is a list containing the number of filters for each convolutional
    layer in the encoder, respectively
        * the filters should be reversed for the decoder
    * latent_dims is a tuple of integers containing the dimensions of the latent
    space representation
    * Each convolution in the encoder should use a kernel size of (3, 3) with same
    padding and relu activation, followed by max pooling of size (2, 2)
    * Each convolution in the decoder, except for the last two, should use a filter
    size of (3, 3) with same padding and relu activation, followed by upsampling of size (2, 2)
        * The second to last convolution should instead use valid padding
        * The last convolution should have the same number of filters as the number
        of channels in input_dims with sigmoid activation and no upsampling
    * Returns: encoder, decoder, auto
        * encoder is the encoder model
        * decoder is the decoder model
        * auto is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    """
    input_img = keras.Input(shape=(input_dims))

    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.Maxpool2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(filters[1], (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Maxpool2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(filters[2], (3, 3), activation='relu', padding='same')(x)
    encoded = keras.layers.Maxpool2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs=input_img, outputs=encoded)

    y = keras.layers.Conv2D(filters[2], (3, 3), activation='relu', padding='same')(encoded)
    y = keras.layers.Upsampling2D((2, 2))(y)
    y = keras.layers.Conv2D(filters[1], (3, 3), activation='relu', padding='valid')(y)
    y = keras.layers.Upsampling2D((2, 2))(y)
    y = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='valid')(y)
    y = keras.layers.Upsampling2D((2, 2))(y)
    decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
    decoder = keras.Model(inputs=input_img, outputs=decoded)

    auto = Model(input_img, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
