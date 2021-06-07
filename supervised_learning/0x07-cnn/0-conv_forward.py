#!/usr/bin/env python3
"""
Function that performs forward propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
        * kh is the filter height
        * kw is the filter width
        * c_prev is the number of channels in the previous layer
        * c_new is the number of channels in the output
    * b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    * activation is an activation function applied to the convolution
    * padding is a string that is either same or valid, indicating the type of padding used
    * stride is a tuple of (sh, sw) containing the strides for the convolution
        * sh is the stride for the height
        * sw is the stride for the width
    * you may import numpy as np
    * Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = np.ceil(((sh * h_prev) - sh + kh - h_prev) // 2)
        pw = np.ceil(((sw * w_prev) - sw + kw - w_prev) // 2)
    elif padding == 'valid':
        ph = 0
        pw = 0

    out_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw) (0, 0)), mode='constant', constant_values=(0, 0))
    out_h = (h_prev - kh + (2 * ph)) // (sh + 1)
    out_w = (w_prev - kw + (2 * pw)) // (sw + 1)
    out_conv = np.zeros((m, out_h, out_w, c_new))

    for z in range(c_new):
        for x in range(out_h):
            for y in range(out_w):
                output[:, y, x, z] = activation((W[:, :, :, z] * out_pad)[
                    :, (sh * y):(sh * y) + kw, (sw * x):(sw * x) + kw].sum(axis=(1, 2, 3))) + b
    return output
