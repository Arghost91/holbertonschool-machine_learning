  
#!/usr/bin/env python3
"""
Function that integrates a polinomy
"""


def poly_integral(poly, C=0):
    """
    Return de derivative of poly, if poly is 
    different to 0
    """
    if (poly==0):
        return None
    else:
        integ = poly.copy()
        i = 0
        for i in range(len(poly)):
            if poly[i]%(i+1)==0:
                integ[i] = int(poly[i]/(i+1))
            else:
                integ[i] = poly[i]/(i+1)
            i += 1
        integ.insert(0,C)
        return integ
