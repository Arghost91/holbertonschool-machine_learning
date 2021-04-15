#!/usr/bin/env python3

# Module that performs matrix multiplication:


def mat_mul(mat1, mat2):
    mat_P = [[0 for fil in range(len(mat2[0]))] for col in range(len(mat1))]
    # With two matrices as input, the function return the multiplication of the
    # two matrices as new matrix if the two matrices have the same shape
    if len(mat1[0]) == len(mat2):
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat1[0])):
                    mat_P[i][j] += mat1[i][k] * mat2[k][j]
        return mat_P
    else:
        return None
    
