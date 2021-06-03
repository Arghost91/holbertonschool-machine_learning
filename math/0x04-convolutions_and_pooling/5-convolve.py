#!/usr/bin/env python3
"""
Function that performs a convolution on grayscale images
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    * images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
        * c is the number of channels in the image
    * kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the kernels for the convolution
        * kh is the height of a kernel
        * kw is the width of a kernel
        * nc is the number of kernels
    * padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        * if ‘same’, performs a same convolution
        * if ‘valid’, performs a valid convolution
        * if a tuple:
            * ph is the padding for the height of the image
            * pw is the padding for the width of the image
        * the image should be padded with 0’s
    * stride is a tuple of (sh, sw)
        * sh is the stride for the height of the image
        * sw is the stride for the width of the image
    * Returns: a numpy.ndarray containing the convolved images
    """
    num = images.shape[0]
    height_im = images.shape[1]
    width_im = images.shape[2]
    c = images.shape[3]

    height_ker = kernels.shape[0]
    width_ker = kernels.shape[1]
    c = kernels.shape[2]
    nc = kernels.shape[3]

    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        output_height = (height_im - height_ker) // sh + 1
        output_width = (width_im - width_ker) // sw + 1
        image_padded = np.copy(images)

    else:
        if padding == 'same':
            padding_h = (((height_im - 1) * sh + height_ker -
                          height_im) // 2) + 1
            padding_w = (((width_im - 1) * sw + width_ker -
                          width_im) // 2) + 1

        else:
            padding_h = padding[0]
            padding_w = padding[1]

        output_height = (height_im - height_ker + (2 * padding_h)) // sh + 1
        output_width = (width_im - width_ker + (2 * padding_w)) // sw + 1
        image_padded = np.zeros((num, output_height, output_width, c))
        image_padded = np.pad(images,
                              ((0, 0), (padding_h, padding_h),
                               (padding_w, padding_w), (0, 0)),
                              mode='constant')
    output = np.zeros((num, output_height, output_width, nc))
    for z in range(nc):
        for x in range(output_width):
            for y in range(output_height):
                output[:, y, x, z] = (kernels[:, :, :, z] * image_padded[
                    :, (sh * y):(sh * y) + height_ker,
                    (sw * x):(sw * x) + width_ker]).sum(axis=(1, 2, 3))
    return output
