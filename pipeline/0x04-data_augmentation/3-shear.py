#!/usr/bin/env python3
"""
function def shear_image(image, intensity): that randomly shears an image
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    * image is a 3D tf.Tensor containing the image to shear
    * intensity is the intensity with which the image should be sheared
    * Returns the sheared image
    """
    shear_im = tf.keras.preprocessing.image.apply_affine_transform(image.numpy(),
                                                                shear=intensity/10)
    return shear_im
