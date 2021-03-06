#!/usr/bin/env python3
"""
Function that calculates the minor matrix of a matrix
"""


def determinant(matrix):
    """
    * matrix is a list of lists whose determinant should be calculated
    * If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    * If matrix is not square, raise a ValueError with the message
    matrix must be a square matrix
    * The list [[]] represents a 0x0 matrix
    * Returns: the determinant of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        diag1 = matrix[0][0] * matrix[1][1]
        diag2 = matrix[0][1] * matrix[1][0]
        return diag1 - diag2
    row = matrix[0]
    det = 0
    cof = 1
    for i in range(len(matrix[0])):
        mat = [l[:] for l in matrix]
        del mat[0]
        for m in mat:
            del m[i]
        det += row[i] * determinant(mat) * cof
        cof = cof * -1
    return det


def minor(matrix):
    """
    * matrix is a list of lists whose minor matrix should be calculated
    * If matrix is not a list of lists, raise a TypeError with
    the message matrix must be a list of lists
    * If matrix is not square or is empty, raise a ValueError with
    the message matrix must be a non-empty square matrix
    * Returns: the minor matrix of matrix
    """
    if type(matrix) is not list or not len(matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a square matrix")
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        return [[1]]
    minor = []
    for i in range(len(matrix)):
        inner = []
        for j in range(len(matrix)):
            mat = [l[:] for l in matrix]
            del mat[i]
            for m in mat:
                del m[j]
            det = determinant(mat)
            inner.append(det)
        minor.append(inner)
    return minor
