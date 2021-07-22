#!/usr/bin/env python3
"""

"""


def determinant(matrix):
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
    for j in matrix:
        if type(j) is not list:
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        diag1 = matrix[0][0] * matrix[1][1]
        diag2 = matrix[0][1] * matrix[1][0]
        return diag1 - diag2
    elif len(matrix) == 3:
        diag1 = matrix[0][0]  *  ((matrix[1][1] * matrix[2][2]) - (matrix[1][2] * matrix[2][1]))
        diag2 = matrix[0][1]  *  ((matrix[1][0] * matrix[2][2]) - (matrix[1][2] * matrix[2][0]))
        diag3 = matrix[0][2]  *  ((matrix[1][0] * matrix[2][1]) - (matrix[1][1] * matrix[2][0]))
        return (diag1 - diag2 + diag3)
