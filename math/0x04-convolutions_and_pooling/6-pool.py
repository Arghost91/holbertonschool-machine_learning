#!/usr/bin/env python3
"""
Function that performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    * images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
        * c is the number of channels in the image
    * kernel_shape is a tuple of (kh, kw) containing the kernel shape for the pooling
        * kh is the height of the kernel
        * kw is the width of the kernel
    * stride is a tuple of (sh, sw)
        * sh is the stride for the height of the image
        * sw is the stride for the width of the image
    * mode indicates the type of pooling
        * max indicates max pooling
        * avg indicates average pooling
    * Returns: a numpy.ndarray containing the pooled images
    """
    num = images.shape[0]
    height_im = images.shape[1]
    width_im = images.shape[2]
    c = images.shape[3]

    height_ker = kernel_shape.shape[0]
    width_ker = kernel_shape.shape[1]
    c = kernel_shape.shape[2]

    sh = stride[0]
    sw = stride[1]

    output_height = (height_im - height_ker) // sh + 1
    output_width = (width_im - width_ker) // sw + 1
    output = np.zeros((num, output_height, output_width, c))

    for x in range(output_width):
        for y in range(output_height):
            image_ = images[:, (sh * y):(sh * y) + height_ker,
                             (sw * x):(sw * x) + width_ker]
            if mode == 'max':
                output[:, y, x] = np.max(img, axis=(1, 2))
            else:
                output[:, y, x] = np.average(img, axis=(1, 2))
    return output
