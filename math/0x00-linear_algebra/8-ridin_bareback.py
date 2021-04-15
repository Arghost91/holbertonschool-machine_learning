#!/usr/bin/env python3

# Module that concatenates two matrices along a specific axis:


def mat_mul(mat1, mat2):
    mat_p = []
    # With two matrices as input, the function return the sum of the two matrices as new list if the two matrices have the same shape 
    if len(mat1[0]) == len(mat2):
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat1[0])):
                    mat_P[i][j] += mat1[i][k] * mat2[k][j]
    return mat_P
  
