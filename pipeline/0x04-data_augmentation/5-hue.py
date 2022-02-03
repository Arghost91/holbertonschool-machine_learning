#!/usr/bin/env python3
"""
function def shear_image(image, intensity): that randomly shears an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    * image is a 3D tf.Tensor containing the image to change
    * delta is the amount the hue should change
    * Returns the altered image
    """
    change_hu = tf.image.adjust_hue(image, delta)
    return change_hu
