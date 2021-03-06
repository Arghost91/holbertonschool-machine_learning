#!/usr/bin/env python3
# Module that concatenates two matrices along a specific axis


def cat_matrices2D(mat1, mat2, axis=0):

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        mat_P = [row.copy() for row in mat1] + mat2.copy()
        return mat_P
    elif axis == 1 and len(mat1) == len(mat2):
        mat_P = [mat1[i] + mat2[i] for i in range(len(mat1))]
        return mat_P
    else:
        return None
