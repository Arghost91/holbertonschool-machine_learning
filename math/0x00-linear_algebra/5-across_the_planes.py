#!/usr/bin/env python3

# Module that adds two matrices element-wise


def add_matrices2D(mat1, mat2):

    """With two matrices as input, the function return the sum of those as new
    matrix if the two matrices have the same shape and if not return None"""

    if len(mat1) != len(mat2):
        return None
    elif len(mat1[0]) != len(mat2[0]):
        return None
    else:
        return [[mat1[i][j] + mat2[i][j]
                 for i in range(len(mat1))] for j in range(len(mat1[0]))]
