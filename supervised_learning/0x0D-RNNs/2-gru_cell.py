#!/usr/bin/env python3
"""
Create the class GRUCell
"""
import numpy as np


class GRUCell:
    """
    Class at represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        * i is the dimensionality of the data
        * h is the dimensionality of the hidden state
        * o is the dimensionality of the outputs
        * Creates the public instance attributes Wz, Wr, Wh, Wy, bz,
        br, bh, by that represent the weights and biases of the cell
        * Wzand bz are for the update gate
        * Wrand br are for the reset gate
        * Whand bh are for the intermediate hidden state
        * Wyand by are for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, 0))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        * x_t is a numpy.ndarray of shape (m, i) that contains the
        data input for the cell
            * m is the batche size for the data
        * h_prev is a numpy.ndarray of shape (m, h) containing the
        previous hidden state
        * The output of the cell should use a softmax activation
        function
        * Returns: h_next, y
            * h_next is the next hidden state
            * y is the output of the cell
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        z = np.matmul(x, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))
        r = np.matmul(x, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))
        x = np.concatenate((r * h_prev, x_t), axis=1)
        h = np.tanh(np.matmul(x, self.Wh) + self.bh)
        h_next = z * h + (1 - z) * h_prev
        y = np.dot(h_next, self.Wy) + self.by
        y = (np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True))
        return h_next, y
