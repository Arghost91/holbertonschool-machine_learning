#!/usr/bin/env python3

import numpy as np

def one_hot_encode(Y, classes):
    
    m = len(Y)
    if type(Y) is not np.ndarray:
        return None
    else:
        one_hot_encode = np.zeros((classes, m))
        one_hot_encode[Y, np.arange(Y.size)] = 1
        return one_hot_encode
