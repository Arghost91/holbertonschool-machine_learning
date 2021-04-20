#!/usr/bin/env python3
"""
Function that returns the cuadratic sum
"""


def summation_i_squared(n):
    """
    Calculate the cuadratic sum for n
    """
    if ((n < 1) or (type(n) is not int)):
        return None
    else:
        calc = (n*(n+1)*(2*n+1))/6
        return (int(calc))
