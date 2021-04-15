#!/usr/bin/env python3
# Module that performs element-wise addition, subtraction, multiplication, and division


def np_elementwise(mat1, mat2):
    sume = mat1 + mat2
    difference = mat1 - mat2
    product = mat1 * mat2
    quotient = mat1 / mat2
    return(sume, difference, product, quotient)
