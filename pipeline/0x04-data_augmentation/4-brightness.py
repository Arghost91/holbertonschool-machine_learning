#!/usr/bin/env python3
"""
function def change_brightness(image, max_delta): that randomly changes the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    * image is a 3D tf.Tensor containing the image to change
    * max_delta is the maximum amount the image should be brightened (or darkened)
    * Returns the altered image
    """
    change_bright = tf.image.adjust_brightness(image, max_delta)
    return change_bright
