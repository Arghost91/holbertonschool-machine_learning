#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
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
    * Returns: a numpy.ndarray containing the convolved images
    """
    num = images[0]
    height_im = images[1]
    width_im = images[2]
    
    height_ker = kernel[0]
    width_ker = kernel[1]
    
    output_height = height_im - height_ker + 1
    output_width = width_im - width_ker + 1
    output = np.zeros((num, output_height, output_width))
    
    for x in range(output_width):
        for y in range(output_height):
            output[:, y, x] = (kernel * images[:, y:y + height_ker,
                                               x:x + width_ker]).sum(axis=(1, 2))
    return output
