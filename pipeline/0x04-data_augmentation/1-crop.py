#!/usr/bin/env python3
"""
function def crop_image(image, size): that performs a random crop of an image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    * image is a 3D tf.Tensor containing the image to crop
    * size is a tuple containing the size of the crop
    * Returns the cropped image
    """
    return tf.random_crop(value=image, size=size)
