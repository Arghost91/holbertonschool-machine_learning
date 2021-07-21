#!/usr/bin/env python3
"""

"""


def determinant(matrix):
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    else:
        if len(matrix) == 1:
            diag1 = matrix[0,0] * matrix[1,1]
            diag2 = matrix[0,1] * matrix[1,0]
            det = diag1 - diag2
        elif len(matrix) == 2:
            diag1 = matrix[0,0]  *  (matrix[1,1] * matrix[2,2] - matrix[1,2] * matrix[2,1])
            diag2 = matrix[0,1]  *  (matrix[0,1] * matrix[2,2] - matrix[1,2] * matrix[0,3])
            diag3 = matrix[0,2]  *  (matrix[0,1] * matrix[1,2] - matrix[2,2] * matrix[0,2])
            det = diag1 - diag2 + diag3
        elif len(matrix) = 0:
            det = 1
    return det
