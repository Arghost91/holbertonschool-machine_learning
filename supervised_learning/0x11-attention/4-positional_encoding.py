#!/usr/bin/env python3
"""
Function that calculates the positional encoding for a transformer
"""
import numpy as nop


def positional_encoding(max_seq_len, dm):
    """
    * max_seq_len is an integer representing the maximum sequence length
    * dm is the model depth
    * Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the positional encoding vectors
    """
    
