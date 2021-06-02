#!/usr/bin/env python3
"""
Function that performs a convolution on grayscale
images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    * images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
    * kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
    * padding is a tuple of (ph, pw)
        * ph is the padding for the height of the image
        * pw is the padding for the width of the image
        * the image should be padded with 0’s
    * Returns: a numpy.ndarray containing the convolved images
    """
    num = images.shape[0]
    height_im = images.shape[1]
    width_im = images.shape[2]
    
    height_ker = kernel.shape[0]
    width_ker = kernel.shape[1]
    
    padding_h = padding[0]
    padding_w = padding[1]

    output_height = height_im - height_ker + (2 * padding_h) + 1
    output_width = width_im - width_ker + (2 * padding_w) + 1
    output = np.zeros((num, output_height, output_width))
    image_padded = np.zeros((num, output_height, output_width))
    image_padded = np.pad(images, ((0,0), (padding_h,padding_h),
                                   (padding_w,padding_w)), mode='constant')
    for x in range(output_width):
        for y in range(output_height):
            output[:, y, x] = (kernel * image_padded[:, y:y + height_ker,
                                               x:x + width_ker]).sum(axis=(1, 2))
    return output
