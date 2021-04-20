#!/usr/bin/env python3
"""
Function that derivates a polinomy
"""


def poly_derivative(poly):
    """
    Return de derivative of poly, if poly is 
    different to 0
    """
    if poly==0:
        return None
    else:
        der = poly.copy()
        der.pop(0)
        i = 1
        for i in range(len(poly)):
            der[i-1] = poly[i]*i
            i += 1
        return der
