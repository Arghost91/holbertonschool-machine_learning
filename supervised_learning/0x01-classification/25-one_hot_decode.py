#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
     
    if type(one_hot) is not np.ndarray:
        return None
    else:
        return np.argmax(one hor, axis=0)
