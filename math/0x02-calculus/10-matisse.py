#!/usr/bin/env python3
"""

"""


def poly_derivative(poly):
    if poly==0:
        return None
    else:
        der = polly.copy()
        i = 1
        for i in range(len(polly)-1):
            der[i] = polly[i]*i
            i += 1
        return der
