#!/usr/bin/env python3
"""
Calculates the adjugate of a matrix
"""


def determinant(matrix):
    """
    * matrix is a list of lists whose determinant should be calculated
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
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    row = matrix[0]
    det = 0
    cofact = 1
    for i in range(len(matrix[0])):
        mat = [l[:] for l in matrix]
        del mat[0]
        for m in mat:
            del m[i]
        det += row[i] * determinant(mat) * cofact
        cofact = cofact * -1
    return det


def cofactor(matrix):
    """
    * matrix is a list of lists whose cofactor matrix should
    be calculated
    * If matrix is not a list of lists, raise a TypeError with
    the message matrix must be a list of lists
    * If matrix is not square or is empty, raise a ValueError
    with the message matrix must be a non-empty square matrix
    * Returns: the cofactor matrix of matrix
    """
    if type(matrix) is not list or not len(matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        return [[1]]
    minor = []
    for i in range(len(matrix)):
        inner = []
        if i % 2 != 0:
            cofact = -1
        else:
            cofact = 1
        for j in range(len(matrix[0])):
            mat = [l[:] for l in matrix]
            del mat[i]
            for m in mat:
                del m[j]
            det = determinant(mat) * cofact
            inner.append(det)
            cofact = cofact * (-1)
        minor.append(inner)
    return minor


def adjugate(matrix):
    """
    a function that calculates the adjugate
    :param matrix: a list of lists whose adjugate matrix should be calculated
    :return: the adjugate of a matrix
    """
    adj = cofactor(matrix)
    transpose = []
    for j in range(len(adj[0])):
        inner = []
        for i in range(len(adj)):
            inner.append(adj[i][j])
        transpose.append(inner)
    return transpose
