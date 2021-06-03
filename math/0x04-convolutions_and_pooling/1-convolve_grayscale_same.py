#!/usr/bin/env python3
"""
Function that performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    * images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
    * kernel is a numpy.ndarray with shape (kh, kw) containing the
    kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
    * if necessary, the image should be padded with 0’s
    * Returns: a numpy.ndarray containing the convolved images
    """
    num = images.shape[0]
    height_im = images.shape[1]
    width_im = images.shape[2]

    height_ker = kernel.shape[0]
    width_ker = kernel.shape[1]

    if height_ker % 2 != 0:
        padding_h = (height_ker - 1) // 2
    else:
        padding_h = height_ker // 2
    if width_ker % 2 != 0:
        padding_w = (width_ker - 1) // 2
    else:
        padding_w = width_ker // 2
    output_height = height_im
    output_width = width_im
    output = np.zeros((num, output_height, output_width))
    image_padded = np.zeros((num, height_im, width_im))
    image_padded = np.pad(images, ((0, 0), (padding_h, padding_h),
                                   (padding_w, padding_w)), mode='constant')
    for x in range(output_width):
        for y in range(output_height):
            output[:, y, x] = (kernel * image_padded[
                :, y:y + height_ker, x:x + width_ker]).sum(axis=(1, 2))
    return output
