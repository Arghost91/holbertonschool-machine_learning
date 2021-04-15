#!/usr/bin/env python3
# Module that concatenates two matrices along a specific axis


def cat_matrices2D(mat1, mat2, axis=0):
    
    if axis == 0:
        mat_P = mat1.copy()
        mat_P += mat2.copy()
        return mat_P
    elif axis == 1:
        for i in range(len(mat2)):
            mat_P = [mat1[i] + mat2[i]]
        return mat_P
    else:
        return None
