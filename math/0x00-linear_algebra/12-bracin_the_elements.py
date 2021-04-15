#!/usr/bin/env python3

# Module that concatenates two matrices along a specific axis:


def np_elementwise(mat1, mat2):
    sum = np.add(mat1, mat2)
    difference = np.substract(mat1, mat2)
    product = np.multiply(mat1, mat2)
    quotient = np.divide(mat1, mat2)
    return(sum, difference, product, quotient)
