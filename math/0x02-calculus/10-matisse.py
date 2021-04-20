#!/usr/bin/env python3
"""

"""


def poly_derivative(poly):
    if poly==0:
        return None
    else:
        der = poly.copy()
        der.pop(0)
        i = 1
        for i in range(len(der)):
            der[i] = poly[i]*i
            i += 1
        return der
