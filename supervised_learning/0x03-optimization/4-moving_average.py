#!/usr/bin/env python3
"""
Function that calculates the weighted moving average of a data set
"""
import numpy as np


def moving_average(data, beta):
    """
    * data is the list of data to calculate the moving average of
    * beta is the weight used for the moving average
    * Your moving average calculation should use bias correction
    * Returns: a list containing the moving averages of data
    """
    val = 0
    wight = []
    for i in range(len(data)):
        val = beta*val + (1 - beta) * data[i]
        wight.append(val / (1 - beta ** (i + 1)))
    return wight
