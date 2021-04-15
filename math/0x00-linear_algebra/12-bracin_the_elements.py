#!/usr/bin/env python3

# Module that concatenates two matrices along a specific axis:


def np_elementwise(mat1, mat2):
    sume = mat1 + mat2
    difference = mat1 - mat2
    product = mat1 * mat2
    quotient = mat1 / mat2
    return(sume, difference, product, quotient)
